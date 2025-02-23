import math
import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from config import *

def get_sinusoidal_embedding(t, dim):
	"""
	Generate sinusoidal positional embeddings.
	t: Tensor of shape (B, 1) containing timesteps.
	Returns: Tensor of shape (B, dim) with sinusoidal embeddings.
	
	Uses torch.einsum to multiply each scalar timestep with a vector of exponents,
	effectively broadcasting the scalar over the embedding dimensions.
	"""
	device = t.device
	half_dim = dim // 2
	emb_scale = math.log(10000) / (half_dim - 1)
	# Compute the exponential factors for half of the embedding dimensions.
	exp_factors = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)  # (half_dim,)
	# Use einsum to multiply each timestep with the exp_factors.
	# t has shape (B, 1) and we want output shape (B, half_dim)
	emb = torch.einsum("bi,j->bij", t, exp_factors).squeeze(1)  # (B, half_dim)
	# Compute sine and cosine embeddings.
	emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
	if dim % 2 == 1:
		emb = F.pad(emb, (0, 1))
	return emb

class FiLMBlock(nn.Module):
	"""
	FiLM block that applies Feature-wise Linear Modulation.
	It computes per-channel scale (gamma) and bias (beta) from a conditioning vector.

	- FiLM layers modulate the feature maps using an external conditioning signal (e.g., 
		image features, state information, and diffusion timestep embeddings).
	- This modulation adjusts the scaling and shifting of feature activations on a per-channel
		basis, allowing the network to dynamically adapt its computations based on the global context.
	- Such conditioning guides the denoising process to ensure that the predicted noise removal 
		aligns with the temporal structure and external conditions.
	"""
	def __init__(self, channels, cond_dim):
		super().__init__()
		self.fc = nn.Linear(cond_dim, channels * 2)  # compute gamma and beta
	
	def forward(self, x, cond):
		# x: (B, channels, T); cond: (B, cond_dim)
		gamma_beta = self.fc(cond)  # (B, channels*2)
		gamma, beta = gamma_beta.chunk(2, dim=-1)  # each: (B, channels)
		# Unsqueeze to prepare for broadcasting.
		gamma = gamma.unsqueeze(-1)  # (B, channels, 1)
		beta = beta.unsqueeze(-1)    # (B, channels, 1)
		# elementwise multiplication: x * (1 + gamma)
		# Use a letter (here "z") instead of "1" for the singleton dimension.
		modulated = torch.einsum("bct, bcz -> bct", x, (1 + gamma))
		return modulated + beta  # FiLM modulation


class DownBlock(nn.Module):
	"""
	Downsampling block for the U-Net encoder.
	Applies a convolution followed by FiLM conditioning and GELU activation.
	"""
	def __init__(self, in_channels, out_channels, cond_dim):
		super().__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
		self.film = FiLMBlock(out_channels, cond_dim)
		self.act = nn.GELU()
	
	def forward(self, x, cond):
		x = self.conv(x)
		x = self.film(x, cond)
		x = self.act(x)
		return x

class UpBlock(nn.Module):
	"""
	Upsampling block for the U-Net decoder.
	It concatenates the upsampled features with the corresponding encoder skip connection,
	then applies convolution, FiLM conditioning, and GELU activation.
	"""
	def __init__(self, in_channels, out_channels, cond_dim):
		super().__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
		self.film = FiLMBlock(out_channels, cond_dim)
		self.act = nn.GELU()
	
	def forward(self, x, skip, cond):
		# Concatenate the upsampled features with the skip connection.
		x = torch.cat([x, skip], dim=1)  # (B, channels, T)
		x = self.conv(x)
		x = self.film(x, cond)
		x = self.act(x)
		return x

class UNet1D(nn.Module):
	"""
	A 1D U-Net for denoising a temporal window of actions.
	The conditioning information (global_cond) is injected into each block via FiLM.

	In our diffusion policy network, we adopt a UNet-based architecture enhanced with 
	Feature-wise Linear Modulation (FiLM). Inspired by the LeRobot implementation.

	Attributes:

	1. Multi-Scale Feature Extraction and Reconstruction:
	   - The UNet architecture is composed of an encoder and a decoder with skip connections,
		 enabling it to capture both global context and local details.
	   - The encoder compresses the input, learning hierarchical representations, while the 
		 decoder reconstructs the signal using skip connections to retain fine-grained information.
	   - This preserves both local temporal variations and global trajectory information.

	2. Effective Global Conditioning via FiLM:
	   - Guides the denoising process to ensure that the predicted noise removal 
		 aligns with the temporal structure and external conditions.

	3. Improved Gradient Flow and Convergence:
	   - The combination of skip connections inherent in UNet and the controlled modulation from FiLM
		 facilitates better gradient propagation during training.
	"""
	def __init__(self, action_dim, cond_dim, hidden_dim=64):
		super().__init__()
		# Encoder: process the input action sequence
		self.initial_conv = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
		self.down1 = DownBlock(hidden_dim, hidden_dim * 2, cond_dim)
		
		# Bottleneck: further processing at the lowest resolution
		self.bottleneck = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
		
		# Decoder: reconstruct the signal and integrate skip connections
		# Note: The skip connection from down1 is concatenated, so input channels = hidden_dim*2 + hidden_dim*2.
		self.up1 = UpBlock(hidden_dim * 2 + hidden_dim * 2, hidden_dim, cond_dim)
		self.final_conv = nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1)
	
	def forward(self, x, cond):
		# x: (B, window_size, action_dim) -> rearranged to (B, action_dim, T)
		x = rearrange(x, 'b t a -> b a t')
		x0 = self.initial_conv(x)         # (B, hidden_dim, T)
		x1 = self.down1(x0, cond)         # (B, hidden_dim*2, T)
		x_b = self.bottleneck(x1)         # (B, hidden_dim*2, T)
		x_up = self.up1(x_b, x1, cond)      # (B, hidden_dim, T)
		out = self.final_conv(x_up)       # (B, action_dim, T)
		out = rearrange(out, 'b a t -> b t a')
		return out

class DiffusionPolicy(nn.Module):
	def __init__(self, action_dim, condition_dim, time_embed_dim=128, window_size=WINDOW_SIZE+1):
		"""
		This network is used in the reverse diffusion process to predict the noise that
		was added to an action sample. In our project, the action is the end-effector (EE)
		position (a 2D vector) and we consider a temporal window of actions. The window
		includes the timestep t-1 as well as t to t+(window_size-1), so the input window length
		is (WINDOW_SIZE+1).

		Parameters:
		- action_dim (int): Dimension of the action (e.g., 2)
		- condition_dim (int): Dimension of the conditioning signal (e.g., 132, which may include image features and states)
		- time_embed_dim (int): Dimension of the sinusoidal embedding for the diffusion timestep.
		- window_size (int): Number of consecutive action timestamps to consider (includes t-1).

		Input:
		- x: a noised action sample of shape (batch, window_size, action_dim)
		- t: the diffusion timestep tensor of shape (batch,)
		- condition: a conditioning signal of shape (batch, condition_dim) (e.g., 132-dim)

		Output:
		- noise_pred: The predicted noise for the entire window with shape (batch, window_size, action_dim)
		"""
		super(DiffusionPolicy, self).__init__()
		self.window_size = window_size
		self.time_embed_dim = time_embed_dim
		
		# Time Embedding:
		# Instead of a learned embedding, we use a sinusoidal embedding (as in transformers)
		# followed by a small MLP to refine the representation.
		self.time_mlp = nn.Sequential(
			nn.Linear(time_embed_dim, time_embed_dim),  # Refine the sinusoidal embedding.
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU()
		)
		
		# Global conditioning is built by concatenating:
		# - The provided conditioning signal (e.g., image features and state information, condition_dim)
		# - The refined time embedding (time_embed_dim)
		# Total global conditioning dimension: condition_dim + time_embed_dim.
		self.global_cond_dim = condition_dim + time_embed_dim
		
		# Main Network: a 1D U-Net that takes the noisy action sequence as input and predicts the noise.
		# The U-Net receives the global conditioning vector at each block.
		self.unet = UNet1D(action_dim, cond_dim=self.global_cond_dim, hidden_dim=64)
	
	def forward(self, x, t, condition):
		"""
		Forward pass through the diffusion policy network.

		Parameters:
		- x: Noised action tensor of shape (batch, window_size, action_dim).
			  This represents a temporal window of EE position samples (including t-1) with added noise.
		- t: Timestep tensor of shape (batch,). Indicates the diffusion timestep for the noise in x.
		- condition: Conditioning signal tensor of shape (batch, condition_dim).
					 Represents desired image features and previous/current states.

		Process:
		1. If x is 2D, add a temporal dimension.
		2. Reshape t to (batch, 1) and convert to float.
		3. Compute a sinusoidal time embedding for t and refine it via an MLP.
		4. Concatenate the refined time embedding with the provided condition to form a global conditioning vector.
		5. Process the noisy action sequence and the global conditioning vector through the U-Net.
		6. Output the predicted noise, reshaped back to (batch, window_size, action_dim).

		Returns:
		- noise_pred: Predicted noise tensor of shape (batch, window_size, action_dim).
		"""
		# If x is 2D, add a temporal dimension.
		if x.dim() == 2:
			x = x.unsqueeze(1)  # x shape becomes (batch, 1, action_dim)

		# Reshape timestep tensor to (batch, 1) and convert to float.
		t = rearrange(t, 'b -> b 1').float()
		# Compute sinusoidal embedding for t (resulting shape: (batch, time_embed_dim))
		t_emb = get_sinusoidal_embedding(t, self.time_embed_dim)
		t_emb = self.time_mlp(t_emb)  # Refine the time embedding

		# Create the global conditioning vector by concatenating the provided condition and the refined time embedding.
		global_cond = torch.cat([condition, t_emb], dim=-1)  # (batch, condition_dim + time_embed_dim)

		# Pass the noisy trajectory and the global conditioning through the U-Net.
		noise_pred = self.unet(x, global_cond)  # (batch, window_size, action_dim)
		return noise_pred

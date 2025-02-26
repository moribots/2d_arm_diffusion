import math
import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from config import *
from visual_encoder import VisualEncoder  # new import for visual encoder

def get_sinusoidal_embedding(t, dim):
	"""
	Generate sinusoidal positional embeddings.
	t: Tensor of shape (B, 1) containing timesteps.
	Returns: Tensor of shape (B, dim) with sinusoidal embeddings.
	"""
	device = t.device
	half_dim = dim // 2
	emb_scale = math.log(10000) / (half_dim - 1)
	exp_factors = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
	emb = torch.einsum("bi,j->bij", t, exp_factors).squeeze(1)
	emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
	if dim % 2 == 1:
		emb = F.pad(emb, (0, 1))
	return emb

class FiLMBlock(nn.Module):
	"""
	FiLM block that applies Feature-wise Linear Modulation.
	"""
	def __init__(self, channels, cond_dim):
		super().__init__()
		self.fc = nn.Linear(cond_dim, channels * 2)
	
	def forward(self, x, cond):
		gamma_beta = self.fc(cond)
		gamma, beta = gamma_beta.chunk(2, dim=-1)
		gamma = gamma.unsqueeze(-1)
		beta = beta.unsqueeze(-1)
		modulated = torch.einsum("bct, bcz -> bct", x, (1 + gamma))
		return modulated + beta

class DownBlock(nn.Module):
	"""
	Downsampling block for the U-Net encoder.
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
	"""
	def __init__(self, in_channels, out_channels, cond_dim):
		super().__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
		self.film = FiLMBlock(out_channels, cond_dim)
		self.act = nn.GELU()
	
	def forward(self, x, skip, cond):
		x = torch.cat([x, skip], dim=1)
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
		self.initial_conv = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
		self.down1 = DownBlock(hidden_dim, hidden_dim * 2, cond_dim)
		self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4, cond_dim)
		self.bottleneck = nn.Conv1d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1)
		self.up2 = UpBlock(hidden_dim * 4 + hidden_dim * 4, hidden_dim * 2, cond_dim)
		self.up1 = UpBlock(hidden_dim * 2 + hidden_dim * 2, hidden_dim, cond_dim)
		self.final_conv = nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1)
	
	def forward(self, x, cond):
		# x shape: (batch, time, action_dim)
		x = rearrange(x, 'b t a -> b a t')  # (batch, channels, time)
		x0 = self.initial_conv(x)            # (batch, hidden_dim, time)
		x1 = self.down1(x0, cond)            # (batch, hidden_dim*2, time)
		x2 = self.down2(x1, cond)            # (batch, hidden_dim*4, time)
		x_b = self.bottleneck(x2)            # (batch, hidden_dim*4, time)
		x_up2 = self.up2(x_b, x2, cond)      # (batch, hidden_dim*2, time)
		x_up1 = self.up1(x_up2, x1, cond)      # (batch, hidden_dim, time)
		out = self.final_conv(x_up1)         # (batch, action_dim, time)
		out = rearrange(out, 'b a t -> b t a')
		return out

class DiffusionPolicy(nn.Module):
	def __init__(self, action_dim, condition_dim, time_embed_dim=128, window_size=WINDOW_SIZE+1):
		"""
		This network is used in the reverse diffusion process to predict the noise that
		was added to an action sample. In our project, the action is the end-effector (EE)
		position (a 2D vector) and we consider a temporal window of actions.
		
		Input:
		- x: a noised action sample of shape (batch, window_size, action_dim)
		- t: the diffusion timestep tensor of shape (batch,)
		- state: Conditioning state tensor of shape (batch, 4)
		- image: Conditioning image tensor of shape (batch, 3, IMG_RES, IMG_RES)
		
		Output:
		- noise_pred: The predicted noise for the entire window with shape (batch, window_size, action_dim)
		"""
		super(DiffusionPolicy, self).__init__()
		self.window_size = window_size
		self.time_embed_dim = time_embed_dim
		
		# Time Embedding:
		self.time_mlp = nn.Sequential(
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU()
		)
		
		# Instantiate the visual encoder to extract image features
		self.visual_encoder = VisualEncoder()  # new visual encoder
		
		# Global conditioning is built by concatenating:
		# - The provided state (4-dim) and image features (32-dim)
		# - The refined time embedding (time_embed_dim)
		# Total global conditioning dimension: (4+32) + time_embed_dim.
		self.global_cond_dim = (CONDITION_DIM - 0) + time_embed_dim  
		
		# Main Network: an upgraded 1D U-Net that takes the noisy action sequence as input and predicts the noise.
		self.unet = UNet1D(action_dim, cond_dim=self.global_cond_dim, hidden_dim=64)
	
	def forward(self, x, t, state, image):
		"""
		Forward pass through the diffusion policy network.

		Parameters:
		- x: Noised action tensor of shape (batch, window_size, action_dim).
		- t: Timestep tensor of shape (batch,).
		- state: Conditioning state tensor of shape (batch, 4).
		- image: Conditioning image tensor of shape (batch, 3, IMG_RES, IMG_RES).

		Returns:
		- noise_pred: Predicted noise tensor of shape (batch, window_size, action_dim).
		"""
		if x.dim() == 2:
			x = x.unsqueeze(1)
		if image.dim() == 3:
			image = image.unsqueeze(0)
		t = rearrange(t, 'b -> b 1').float()
		t_emb = get_sinusoidal_embedding(t, self.time_embed_dim)
		t_emb = self.time_mlp(t_emb)
		
		# Extract image features using the visual encoder
		image_features = self.visual_encoder(image)  # (batch, 32)
		# Concatenate state and image features to form combined condition
		combined_condition = torch.cat([state, image_features], dim=-1)  # (batch, 4+32)
		
		# Create the global conditioning vector by concatenating combined_condition and refined time embedding.
		global_cond = torch.cat([combined_condition, t_emb], dim=-1)  # (batch, (4+32)+time_embed_dim)
		
		noise_pred = self.unet(x, global_cond)
		return noise_pred

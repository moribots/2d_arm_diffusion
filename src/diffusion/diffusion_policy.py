"""
This module implements the diffusion policy network used in the reverse diffusion process.
It predicts the noise added to an action sequence given conditioning information.
Includes:
  - A diffusion step encoder with a sinusoidal positional embedding.
  - A configurable Conditional U-Net architecture.
  - Dedicated downsampling and upsampling modules.
  - Global conditioning combining agent state, visual features, and timestep embedding.
  - A FiLM module that outputs both scale and bias parameters.
  
Overall Architecture:
  
		  +-------------------------------+
		  | Input: x, t, state, image     |
		  +-------------------------------+
					  |
					  v
		  +-------------------------------+
		  |   Visual Encoder              | <-- processes image -> image_features
		  +-------------------------------+
					  |
					  v
		  +-------------------------------+
		  | Global Conditioning           | <-- concatenates state & image_features
		  +-------------------------------+
					  |
					  v
		  +-------------------------------+
		  | Diffusion Step Encoder        | <-- encodes timestep t
		  +-------------------------------+
					  |
					  v
		  +-------------------------------+
		  | Conditional U-Net             | <-- predicts noise from input and conditioning
		  +-------------------------------+
					  |
					  v
		  +-------------------------------+
		  | Predicted Noise Output        |
		  +-------------------------------+
"""

import math
import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from src.config import *
from src.diffusion.visual_encoder import VisualEncoder
from src.seed import set_seed

# Set the random seed for reproducibility.
set_seed(42)


class SinusoidalPosEmb(nn.Module):
	"""
	Computes sinusoidal positional embeddings for diffusion timesteps.

	This module generates embeddings that encode the diffusion timestep using sine and cosine functions
	with different frequencies.

	Args:
		dim (int): Dimensionality of the embedding.
	"""
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Generate sinusoidal embeddings for the input timesteps.

		Args:
			x (torch.Tensor): Tensor of shape (B,) containing diffusion timesteps.
		
		Returns:
			torch.Tensor: Sinusoidal embeddings of shape (B, dim).
		"""
		device = x.device
		half_dim = self.dim // 2
		emb_scale = math.log(10000) / (half_dim - 1)
		# Compute exponential factors for the positional encoding.
		exp_factors = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
		# Compute the outer product between timesteps and the exponential factors.
		emb = torch.einsum('b,d->bd', x, exp_factors)
		# Concatenate sine and cosine embeddings.
		emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
		return emb


class DiffusionStepEncoder(nn.Module):
	"""
	Encodes the diffusion timestep into an embedding using a sinusoidal positional embedding.

	The encoder first applies a sinusoidal embedding followed by a two-layer MLP with a 4× expansion.
	
	Args:
		embed_dim (int): Dimensionality of the diffusion step embedding.
	"""
	def __init__(self, embed_dim: int):
		super().__init__()
		self.embed_dim = embed_dim
		self.encoder = nn.Sequential(
			SinusoidalPosEmb(embed_dim),
			nn.Linear(embed_dim, embed_dim * 4),
			nn.Mish(),
			nn.Linear(embed_dim * 4, embed_dim)
		)

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		"""
		Encode the diffusion timestep.

		Args:
			t (torch.Tensor): Tensor of shape (B,) containing diffusion timesteps.
		
		Returns:
			torch.Tensor: Encoded diffusion step embedding of shape (B, embed_dim).
		"""
		return self.encoder(t)


class Downsample1d(nn.Module):
	"""
	Downsampling module using a strided 1D convolution.

	This module reduces the temporal resolution of the feature maps.
	
	Args:
		dim (int): Number of channels in the input (and output).
	"""
	def __init__(self, dim: int):
		super().__init__()
		# Kernel=3, stride=2, padding=1 guarantees that for appropriately chosen input lengths,
		# the temporal dimension reduces as: L_out = floor((L_in + 2*padding - kernel_size)/stride + 1)
		self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.conv(x)


class Upsample1d(nn.Module):
	"""
	Upsampling module using a transposed 1D convolution.

	This module increases the temporal resolution of the feature maps.
	
	Args:
		dim (int): Number of channels in the input (and output).
	"""
	def __init__(self, dim: int):
		super().__init__()
		# Kernel=4, stride=2, padding=1 guarantees that the output length is exactly doubled.
		self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.conv(x)


class FiLM(nn.Module):
	"""
	FiLM conditioning module that produces per-channel scale and bias parameters.

	This module takes a conditioning vector and outputs two modulation parameters per channel
	(scale and bias), which are applied multiplicatively and additively, respectively, to the input feature map.

	This conditioning technique adaptively modifies the feature maps based on conditioning information,
	This is critical to integrating varied sources of information (e.g., diffusion step and global state) into the network.

	Args:
		channels (int): Number of channels in the input feature map.
		cond_dim (int): Dimensionality of the conditioning vector.
	"""
	def __init__(self, channels: int, cond_dim: int):
		super().__init__()
		self.fc = nn.Linear(cond_dim, channels * 2)

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		"""
		Apply FiLM modulation to the feature map.

		Args:
			x (torch.Tensor): Feature map of shape (B, channels, T).
			cond (torch.Tensor): Conditioning tensor of shape (B, cond_dim).
		
		Returns:
			torch.Tensor: Modulated feature map of shape (B, channels, T).
		"""
		# Produce modulation parameters; expected shape: (B, channels * 2)
		mod_params = self.fc(cond)
		# Correctly reshape to (B, 2, channels, 1) such that:
		#   - The first half along the second dimension corresponds to scale.
		#   - The second half corresponds to bias.
		mod_params = rearrange(mod_params, 'b (d c) -> b d c 1', d=2)
		scale, bias = mod_params[:, 0], mod_params[:, 1]
		return x * scale + bias


class ConditionalResidualBlock1D(nn.Module):
	"""
	Residual block for 1D diffusion with conditional FiLM modulation.

	This block consists of:
	  - A first Conv1d layer followed by GroupNorm and Mish activation.
	  - FiLM conditioning that applies scale and bias modulation.
	  - A second Conv1d layer followed by GroupNorm and Mish activation.
	  - A residual connection (with a possible 1×1 convolution if channel dimensions differ).
	
	Diagram:
		 x ──► Conv1d ─► GN ─► Mish ─► FiLM ─► Conv1d ─► GN ─► Mish
		  │                                          │
		  └───────────── [1x1 Projection] ───────────► +

	The Conv1d layers transform feature dimensions from in_channels to out_channels while 
	maintaining the temporal dimension. Additionally, it maintains the temporal structure of
	the input as well as its order (e.g the relative relevance of each input)
	The FiLM module injects contextual information (e.g diffusion time step).
	The residual connection helps information flow and stabilizes training by adding the input to the output.

	Args:
		in_channels (int): Number of input channels.
		out_channels (int): Number of output channels.
		cond_dim (int): Dimensionality of the conditioning vector.
		kernel_size (int, optional): Convolution kernel size. Default is 3.
		n_groups (int, optional): Number of groups for GroupNorm. Default is 8.
	"""
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
		super().__init__()
		padding = kernel_size // 2
		# First convolutional block.
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
		self.gn1 = nn.GroupNorm(n_groups, out_channels)
		self.act1 = nn.Mish()
		# FiLM conditioning for modulation.
		self.film = FiLM(out_channels, cond_dim)
		# Second convolutional block.
		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
		self.gn2 = nn.GroupNorm(n_groups, out_channels)
		self.act2 = nn.Mish()
		# Residual connection (projection if in_channels != out_channels).
		self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for the conditional residual block.

		Args:
			x (torch.Tensor): Input tensor of shape (B, in_channels, T).
			cond (torch.Tensor): Conditioning tensor of shape (B, cond_dim).
		
		Returns:
			torch.Tensor: Output tensor of shape (B, out_channels, T) after residual addition.
		"""
		residual = self.res_conv(x)
		# First convolutional block.
		out = self.conv1(x)
		out = self.gn1(out)
		out = self.act1(out)
		# Apply FiLM modulation.
		out = self.film(out, cond)
		# Second convolutional block.
		out = self.conv2(out)
		out = self.gn2(out)
		out = self.act2(out)
		return out + residual


class UNet1D(nn.Module):
	"""
	A configurable 1D U-Net for noise prediction in diffusion processes.

	This architecture incorporates:
	  - A diffusion step encoder to generate a time embedding.
	  - Multiple downsampling stages with conditional residual blocks.
	  - A bottleneck consisting of additional residual blocks.
	  - Corresponding upsampling stages with skip connections.
	
	The conditioning used in each residual block is a concatenation of the diffusion step embedding and the global conditioning.

	Note:
		For this network to guarantee matching temporal dimensions without interpolation,
		the input sequence length must be chosen appropriately (e.g., even lengths when using these down/up sampling parameters).

	Args:
		input_dim (int): Dimensionality of the input (e.g., action space dimension).
		global_cond_dim (int): Dimensionality of the global conditioning vector.
		diffusion_step_embed_dim (int, optional): Embedding dimension for the diffusion step. Default is 256.
		down_dims (list, optional): List of channel dimensions for each encoder level. Default is [256, 512, 1024].
		kernel_size (int, optional): Convolution kernel size. Default is 5.
		n_groups (int, optional): Number of groups for GroupNorm. Default is 8.
	"""
	def __init__(self,
				 input_dim: int,
				 global_cond_dim: int,
				 diffusion_step_embed_dim: int = 256,
				 down_dims: list = [256, 512, 1024],
				 kernel_size: int = 5,
				 n_groups: int = 8):
		super().__init__()
		# Diffusion step encoder to encode the timestep.
		self.diffusion_step_encoder = DiffusionStepEncoder(diffusion_step_embed_dim)
		# Combined conditioning: diffusion step embedding concatenated with global conditioning.
		cond_dim = diffusion_step_embed_dim + global_cond_dim

		# Initial convolution to project input to the first channel dimension.
		self.initial_conv = nn.Conv1d(input_dim, down_dims[0], kernel_size=3, padding=1)

		# Build encoder modules.
		self.down_modules = nn.ModuleList()
		dims = [down_dims[0]] + down_dims[1:]
		for i, out_ch in enumerate(dims):
			in_ch = dims[i - 1] if i > 0 else down_dims[0]
			# Each down module: two conditional residual blocks and a downsampling operation (except at the last level).
			block1 = ConditionalResidualBlock1D(in_channels=in_ch, out_channels=out_ch, cond_dim=cond_dim,
												  kernel_size=kernel_size, n_groups=n_groups)
			block2 = ConditionalResidualBlock1D(in_channels=out_ch, out_channels=out_ch, cond_dim=cond_dim,
												  kernel_size=kernel_size, n_groups=n_groups)
			downsample = Downsample1d(out_ch) if i < len(dims) - 1 else nn.Identity()
			self.down_modules.append(nn.ModuleList([block1, block2, downsample]))

		# Bottleneck: two conditional residual blocks at the deepest level.
		self.bottleneck = nn.Sequential(
			ConditionalResidualBlock1D(in_channels=dims[-1], out_channels=dims[-1], cond_dim=cond_dim,
									   kernel_size=kernel_size, n_groups=n_groups),
			ConditionalResidualBlock1D(in_channels=dims[-1], out_channels=dims[-1], cond_dim=cond_dim,
									   kernel_size=kernel_size, n_groups=n_groups)
		)

		# Build decoder (upsampling) modules.
		self.up_modules = nn.ModuleList()
		x_channels = dims[-1]  # channels from bottleneck output
		# Iterate over encoder stages in reverse order (using stored skip connections)
		for i in range(len(dims) - 1):
			skip_channels = dims[-1 - i]  # corresponding skip connection channels
			in_ch = x_channels + skip_channels  # sum of current x channels and skip connection channels
			out_ch = dims[-2 - i]  # target channel count for this stage
			block1 = ConditionalResidualBlock1D(
				in_channels=in_ch, 
				out_channels=out_ch, 
				cond_dim=cond_dim,
				kernel_size=kernel_size, 
				n_groups=n_groups
			)
			block2 = ConditionalResidualBlock1D(
				in_channels=out_ch, 
				out_channels=out_ch, 
				cond_dim=cond_dim,
				kernel_size=kernel_size, 
				n_groups=n_groups
			)
			upsample = Upsample1d(out_ch)
			self.up_modules.append(nn.ModuleList([block1, block2, upsample]))
			x_channels = out_ch  # update for next stage

		# Final convolution to project back to the input dimension.
		self.final_conv = nn.Conv1d(down_dims[0], input_dim, kernel_size=3, padding=1)

	def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the U-Net.

		Args:
			sample (torch.Tensor): Noised action sequence of shape (B, T, input_dim).
			timestep (torch.Tensor): Diffusion timestep tensor of shape (B,).
			global_cond (torch.Tensor): Global conditioning tensor (e.g., state and visual features) of shape (B, global_cond_dim).
		
		Returns:
			torch.Tensor: Predicted noise of shape (B, T, input_dim).

		Note:
			This implementation assumes that the sequence lengths match exactly
			after downsampling and upsampling, so no interpolation is performed.
		"""
		# Rearrange input from (B, T, input_dim) to (B, input_dim, T)
		x = rearrange(sample, 'b t c -> b c t')
		original_seq_len = x.shape[2]

		# Process diffusion timestep.
		if not torch.is_tensor(timestep):
			timestep = torch.tensor([timestep], dtype=torch.float, device=x.device)
		elif torch.is_tensor(timestep) and timestep.dim() == 0:
			timestep = timestep.unsqueeze(0)
		timestep = timestep.float()
		# Encode the diffusion timestep.
		t_emb = self.diffusion_step_encoder(timestep)
		# Concatenate diffusion step embedding with global conditioning.
		cond = torch.cat([t_emb, global_cond], dim=-1)

		# Initial convolution.
		x = self.initial_conv(x)

		# Encoder forward pass with skip connections.
		skips = []
		for block1, block2, downsample in self.down_modules:
			x = block1(x, cond)
			x = block2(x, cond)
			skips.append(x)
			x = downsample(x)

		# Bottleneck.
		for layer in self.bottleneck:
			x = layer(x, cond)

		# Decoder forward pass using skip connections.
		for block1, block2, upsample in self.up_modules:
			skip = skips.pop()
			# Directly concatenate since the design guarantees matching temporal dimensions.
			x = torch.cat([x, skip], dim=1)
			x = block1(x, cond)
			x = block2(x, cond)
			x = upsample(x)

		# Final projection.
		x = self.final_conv(x)
		# Rearrange back to (B, T, input_dim).
		x = rearrange(x, 'b c t -> b t c')
		return x


class DiffusionPolicy(nn.Module):
	"""
	DiffusionPolicy predicts the noise added to an action sequence during the diffusion process.

	It conditions on:
	  - Agent state (e.g., from t-1 and t).
	  - Visual features extracted from images.
	  - Diffusion timestep via a diffusion step encoder.
	
	Global conditioning is constructed by concatenating the agent state and visual features.
	This global conditioning is then combined with the diffusion step embedding within the U-Net.
	
	Args:
		action_dim (int, optional): Dimensionality of the action space.
		condition_dim (int, optional): Dimensionality of the agent state.
		window_size (int, optional): Length of the action sequence.
	"""
	def __init__(self, action_dim=ACTION_DIM, condition_dim=CONDITION_DIM, window_size=WINDOW_SIZE):
		super(DiffusionPolicy, self).__init__()
		self.window_size = int(window_size)

		# Visual encoder for extracting image features.
		self.visual_encoder = VisualEncoder()

		# Global conditioning from state and image features.
		# Assumes two states (e.g., from t-1 and t) and two images.
		combined_dim = 2 * (condition_dim + IMAGE_FEATURE_DIM)
		self.global_cond_dim = combined_dim

		# Use the deeper UNet1D with the diffusion step encoder.
		self.unet = UNet1D(input_dim=action_dim,
						   global_cond_dim=self.global_cond_dim,
						   diffusion_step_embed_dim=128,
						   down_dims=[128, 256, 512],
						   kernel_size=5,
						   n_groups=8)

	def forward(self, x, t, state, image):
		"""
		Forward pass of the diffusion policy network.

		Args:
			x (torch.Tensor): Noised action sequence of shape (B, window_size, action_dim).
			t (torch.Tensor): Diffusion timestep tensor of shape (B,).
			state (torch.Tensor): Agent state of shape (B, state_dim).
			image (torch.Tensor or list/tuple): A single image or a pair of images (each of shape (B, 3, IMG_RES, IMG_RES)).
		
		Returns:
			torch.Tensor: Predicted noise of shape (B, window_size, action_dim).
		"""
		# Ensure x has shape (B, window_size, action_dim).
		if x.dim() == 2:
			x = x.unsqueeze(1)

		# Process image input:
		# If image is provided as a list or tuple of two images, use them directly.
		if isinstance(image, (list, tuple)) and len(image) == 2:
			img_feat0 = self.visual_encoder(image[0])
			img_feat1 = self.visual_encoder(image[1])
			image_features = torch.cat([img_feat0, img_feat1], dim=-1)
		# Handle the case where image is a tensor with shape (B, 2, 3, IMG_RES, IMG_RES)
		elif image.dim() == 5 and image.size(1) == 2:
			img_feat0 = self.visual_encoder(image[:, 0])
			img_feat1 = self.visual_encoder(image[:, 1])
			image_features = torch.cat([img_feat0, img_feat1], dim=-1)
		else:
			# If image is a single image tensor, ensure it has shape (B, 3, IMG_RES, IMG_RES)
			if image.dim() == 3:
				image = image.unsqueeze(0)
			feat = self.visual_encoder(image)
			# Duplicate features if needed
			image_features = rearrange(repeat(feat, 'b d -> b (repeat d)', repeat=2), 'b d -> b d')

		# Combine agent state and image features for global conditioning.
		global_cond = torch.cat([state, image_features], dim=-1)
		# Predict noise using the UNet1D.
		noise_pred = self.unet(x, t, global_cond)
		return noise_pred

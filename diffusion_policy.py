"""
This module implements the diffusion policy network used in the reverse diffusion process.
It predicts the noise added to an action sequence given conditioning information.
Contains:
  - A time embedding module.
  - A multi-residual block U-Net architecture.
  - Scale and bias modulation via FiLM,	with group normalization and Mish activation.

Overall Architecture:
  
		   +------------------+
		   |  Input: t, state |
		   +------------------+
					|
					v
		 +----------------------+
		 | Sinusoidal Embedding | <--- DiffusionTimeEmbedding
		 +----------------------+
					|
					v
		  +-------------------+
		  |       MLP         |
		  | (Linear, Mish,    |
		  |   Linear)         |
		  +-------------------+
					|
					v
			 Time Embedding (t_emb)
					|
					v
		   +--------------------+
		   | Visual Encoder     | <-- processes image -> image_features
		   +--------------------+
					|
					v
		  +----------------------+
		  | Concatenate state,   |
		  | image_features, t_emb|  --> global_cond
		  +----------------------+
					|
					v
		  +----------------------+
		  |        U-Net         | <-- UNet1D predicts noise
		  +----------------------+
					|
					v
			Predicted noise output
"""

import math
import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from config import *
from visual_encoder import VisualEncoder
# Set random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)

def get_sinusoidal_embedding(t, dim):
	"""
	Generate sinusoidal positional embeddings for diffusion timesteps.

	Args:
		t (Tensor): Tensor of shape (B, 1) containing timesteps.
		dim (int): Dimension of the embedding.

	Returns:
		Tensor: Sinusoidal embeddings of shape (B, dim).
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

class DiffusionTimeEmbedding(nn.Module):
	"""
	Dedicated time embedding module.
	
	First computes a sinusoidal embedding for the input timesteps and then passes it through
	an MLP with an expansion factor (x4) followed by Mish activations. This yields a richer
	and more expressive embedding for the diffusion timestep.
	"""
	def __init__(self, embed_dim: int):
		super().__init__()
		self.embed_dim = embed_dim
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, embed_dim * 4),
			nn.Mish(),  # Smooth, non-monotonic activation that helps reduce vanishing gradients and improve learning
			nn.Linear(embed_dim * 4, embed_dim),
		)

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			t (Tensor): Tensor of shape (B, 1) containing timesteps.
		Returns:
			Tensor: Processed time embedding of shape (B, embed_dim).
		"""
		# Get sinusoidal embedding.
		sinusoid = get_sinusoidal_embedding(t, self.embed_dim)
		# Pass through MLP.
		return self.mlp(sinusoid)

class FiLM(nn.Module):
	"""
	FiLM conditioning module that outputs bias (and optionally scale) modulation parameters.
	
	This layer takes a conditioning vector and produces per-channel modulation parameters.
	"""
	def __init__(self, channels: int, cond_dim: int):
		super().__init__()
		# Produce only bias modulation parameters.
		self.fc = nn.Linear(cond_dim, channels)

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		# Compute bias modulation and add to the input feature map.
		bias = self.fc(cond).unsqueeze(-1)  # Shape: (B, channels, 1)
		return x + bias

class ResidualBlock1D(nn.Module):
	"""
	Residual block for 1D diffusion.

	This block comprises two convolutional layers, each followed by GroupNorm and
	Mish activation. In between the convolutions, FiLM conditioning is applied.
	A residual connection (with a possible 1x1 convolution) is added to aid training.

	Diagram:
		 x (input)
		   |
		 Conv1D + GN + Mish
		   |
		 FiLM Conditioning
		   |
		 Conv1D + GN + Mish
		   |     
	  +----Residual (x or x projected)----+
		   |
		 Output

	Attributes:
		conv1 (nn.Conv1d): First convolutional layer.
		gn1 (nn.GroupNorm): Group normalization after conv1.
		act1 (nn.Mish): Mish activation after first conv.
		film (FiLM): FiLM module for conditioning.
		conv2 (nn.Conv1d): Second convolutional layer.
		gn2 (nn.GroupNorm): Group normalization after conv2.
		act2 (nn.Mish): Mish activation after second conv.
		res_conv (nn.Conv1d or nn.Identity): Residual connection (projection if needed).
	"""
	def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
		super().__init__()
		padding = kernel_size // 2
		# First convolutional layer.
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
		self.gn1 = nn.GroupNorm(n_groups, out_channels)
		self.act1 = nn.Mish()
		# FiLM for conditioning.
		self.film = FiLM(out_channels, cond_dim)
		# Second convolutional layer.
		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
		self.gn2 = nn.GroupNorm(n_groups, out_channels)
		self.act2 = nn.Mish()
		# Residual connection (1x1 conv if necessary).
		self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (Tensor): Input tensor of shape (B, in_channels, T).
			cond (Tensor): Conditioning tensor of shape (B, cond_dim).
		Returns:
			Tensor: Output tensor of shape (B, out_channels, T).
		"""
		residual = self.res_conv(x)
		out = self.conv1(x)
		out = self.gn1(out)
		out = self.act1(out)
		# Apply FiLM conditioning.
		out = self.film(out, cond)
		out = self.conv2(out)
		out = self.gn2(out)
		out = self.act2(out)
		return out + residual

class UNet1D(nn.Module):
	"""
	A simplified 1D U-Net for denoising temporal action sequences for PushT task.
	This lighter architecture has two encoder/decoder stages instead of three,
	and uses smaller hidden dimensions appropriate for the simpler PushT task.
	
	Architecture Diagram:
		Input (B, T, action_dim)
				|
		[Rearrange: (B, action_dim, T)]
				|
		 Initial Conv (Conv1d)
				|
		 Encoder Stage 1:
			- 2 x ResidualBlock1D (hidden_dim)
			- Save skip1, then Downsample (Conv1d, stride=2) -> hidden_dim * 2
				|
		 Encoder Stage 2:
			- 2 x ResidualBlock1D (hidden_dim * 2)
			- Save skip2, then Downsample (Conv1d, stride=2) -> hidden_dim * 2
				|
		 Bottleneck:
			- 2 x ResidualBlock1D (hidden_dim * 2)
				|
		 Decoder Stage 2:
			- Upsample (ConvTranspose1d)
			- Concatenate skip2
			- 2 x ResidualBlock1D (reduce channels to hidden_dim * 2)
				|
		 Decoder Stage 1:
			- Upsample (ConvTranspose1d)
			- Concatenate skip1
			- 2 x ResidualBlock1D (reduce channels to hidden_dim)
				|
		 Final Conv (Conv1d) to project to action_dim
				|
		 [Rearrange back to (B, T, action_dim)]
				|
		 Output

	Args:
		action_dim (int): Dimensionality of the action space.
		cond_dim (int): Dimensionality of the conditioning vector.
		hidden_dim (int): Base hidden dimension (default: 512).
	"""
	def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int = 512):
		super().__init__()
		# Initial convolution: action_dim -> hidden_dim
		self.initial_conv = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
		
		# Encoder Stage 1: 2 blocks at 64 channels, then downsample with stride=2 (channels remain 64).
		self.enc1_block1 = ResidualBlock1D(hidden_dim, hidden_dim, cond_dim)
		self.enc1_block2 = ResidualBlock1D(hidden_dim, hidden_dim, cond_dim)
		 # Convert to 128 channels.
		self.down1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
		
		# Encoder Stage 2
		self.enc2_block1 = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim)
		self.enc2_block2 = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim)
		
		# Bottleneck
		self.bottleneck_block1 = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim)
		self.bottleneck_block2 = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim)
		
		# Decoder Stage 2 - Use transposed conv with parameters that exactly undo the downsampling
		# kernel_size=4 and stride=2 with padding=1 and output_padding=0 will exactly double the temporal dimension
		self.up2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim * 2, 
									  kernel_size=4, stride=2, padding=1, output_padding=0)
		# Skip from stage 2 the name number of channels, so we double with concatenation.
		self.dec2_block1 = ResidualBlock1D(hidden_dim * 2 * 2, hidden_dim, cond_dim)
		self.dec2_block2 = ResidualBlock1D(hidden_dim, hidden_dim, cond_dim)
		
		# Decoder Stage 1 - Same parameters to match dimensions exactly
		self.up1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 
									 kernel_size=4, stride=2, padding=1, output_padding=0)
		# Residual from stage 1 has 64 channels; concatenation yields 128 channels.
		self.dec1_block1 = ResidualBlock1D(hidden_dim + hidden_dim, hidden_dim, cond_dim)
		self.dec1_block2 = ResidualBlock1D(hidden_dim, hidden_dim, cond_dim)
		
		# Final projection
		self.final_conv = nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1)

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (Tensor): Input tensor of shape (B, T, action_dim).
			cond (Tensor): Global conditioning tensor of shape (B, cond_dim).
		Returns:
			Tensor: Output tensor of shape (B, T, action_dim).
		"""
		# Rearrange input to (B, action_dim, T)
		x = rearrange(x, 'b t a -> b a t')
		
		# Store original sequence length to validate output
		original_seq_len = x.shape[2]
		
		# Initial convolution
		x0 = self.initial_conv(x)
		
		# Encoder Stage 1
		x1 = self.enc1_block1(x0, cond)
		x1 = self.enc1_block2(x1, cond)
		skip1 = x1  # Skip from stage 1 (channels: hidden_dim)
		x1_down = self.down1(x1)

		# Encoder Stage 2
		x2 = self.enc2_block1(x1_down, cond)
		x2 = self.enc2_block2(x2, cond)
		skip2 = x2
		
		# Bottleneck
		xb = self.bottleneck_block1(x2, cond)
		xb = self.bottleneck_block2(xb, cond)
		
		# Decoder Stage 2 - The up2 operation should exactly match skip2's temporal dimension
		x_up2 = self.up2(xb)
		
		# If shapes don't match, use interpolation to safely resize without losing information
		# This follows the approach from the original U-Net paper (Ronneberger et al., 2015)
		# where features are resized to match exactly for concatenation with skip connections
		if x_up2.shape[2] != skip2.shape[2]:
			# mode='linear': Uses linear interpolation appropriate for 1D sequences
			# - For action sequences, linear maintains the smoothness of motion
			# align_corners=True: Makes sure the corner pixels of input and output are aligned
			# - Critical for action sequences where endpoints contain important poses
			x_up2 = F.interpolate(x_up2, size=skip2.shape[2], mode='linear', align_corners=True)
			
		x_up2 = torch.cat([x_up2, skip2], dim=1)
		x_up2 = self.dec2_block1(x_up2, cond)
		x_up2 = self.dec2_block2(x_up2, cond)
		
		# Decoder Stage 1 - The up1 operation should exactly match skip1's temporal dimension
		x_up1 = self.up1(x_up2)
		
		# If shapes don't match, use interpolation to safely resize without losing information
		if x_up1.shape[2] != skip1.shape[2]:
			# Using same interpolation settings as above for consistency
			# Ref: "U-Net: Convolutional Networks for Biomedical Image Segmentation" 
			# Ronneberger et al., MICCAI 2015
			x_up1 = F.interpolate(x_up1, size=skip1.shape[2], mode='linear', align_corners=True)
			
		x_up1 = torch.cat([x_up1, skip1], dim=1)
		x_up1 = self.dec1_block1(x_up1, cond)
		x_up1 = self.dec1_block2(x_up1, cond)
		
		# Final projection and rearrange output back to (B, T, action_dim)
		out = self.final_conv(x_up1)
		
		# Ensure output has the same sequence length as input
		if out.shape[2] != original_seq_len:
			# Final interpolation to match input dimensions exactly
			# Linear interpolation preserves the continuous nature of action sequences
			# while maintaining endpoint values with align_corners=True
			out = F.interpolate(out, size=original_seq_len, mode='linear', align_corners=True)
		
		out = rearrange(out, 'b a t -> b t a')
		return out

class DiffusionPolicy(nn.Module):
	"""
	DiffusionPolicy predicts the noise added to an action sequence during the diffusion process.
	
	It conditions on:
	  - Agent state (4 numbers from t-1 and t).
	  - Visual features from two images (each produces 64 features, total 128).
	  - Diffusion timestep embedding (128 dimensions) from a dedicated time embedding module.
	  
	Global conditioning dimension: 4 + 128 + 128 = 260.
	"""
	def __init__(self, action_dim=ACTION_DIM, condition_dim=CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE+1):
		super(DiffusionPolicy, self).__init__()
		self.window_size = int(window_size)
		self.time_embed_dim = time_embed_dim

		# Dedicated time embedding module.
		self.time_embedding = DiffusionTimeEmbedding(time_embed_dim)

		# Visual encoder to extract features from images.
		self.visual_encoder = VisualEncoder()

		# Combined conditioning: two states (2 * 2) + two images (2 * IMAGE_FEATURE_DIM).
		# 2 comes from our use of t-1 and t.
		combined_dim = 2 * (CONDITION_DIM + IMAGE_FEATURE_DIM)
		self.global_cond_dim = combined_dim + time_embed_dim

		# U-Net with two down/up sampling stages with skip connections for noise prediction.
		self.unet = UNet1D(action_dim=int(action_dim), cond_dim=self.global_cond_dim, hidden_dim=256)

	def forward(self, x, t, state, image):
		"""
		Forward pass of the DiffusionPolicy.
		
		Args:
			x (Tensor): Noised action sequence of shape (B, window_size, action_dim).
			t (Tensor): Diffusion timestep (B,).
			state (Tensor): Agent state of shape (B, 4).
			image (Tensor or list/tuple): Either a single image or a pair [img_t-1, img_t],
										  each of shape (B, 3, IMG_RES, IMG_RES).
										  
		Returns:
			Tensor: Predicted noise of shape (B, window_size, action_dim).
		"""
		if x.dim() == 2:
			x = x.unsqueeze(1)
		# Process image input: assume a pair is provided.
		if isinstance(image, (list, tuple)) and len(image) == 2:
			img_feat0 = self.visual_encoder(image[0])
			img_feat1 = self.visual_encoder(image[1])
			image_features = torch.cat([img_feat0, img_feat1], dim=-1)  # (B, 128)
		else:
			if image.dim() == 3:
				image = image.unsqueeze(0)
			feat = self.visual_encoder(image)
			image_features = torch.cat([feat, feat], dim=-1)
		# Process diffusion timestep using dedicated time embedding.
		t = t.view(-1, 1).float()
		t_emb = self.time_embedding(t)
		# Combine agent state and image features.
		combined_condition = torch.cat([state, image_features], dim=-1)  # (B, 132)
		global_cond = torch.cat([combined_condition, t_emb], dim=-1)  # (B, 260)
		noise_pred = self.unet(x, global_cond)
		return noise_pred

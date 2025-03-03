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
	FiLM conditioning module that outputs both scale and bias parameters.
	
	This layer takes a conditioning vector and produces per-channel modulation parameters.
	"""
	def __init__(self, channels: int, cond_dim: int):
		super().__init__()
		# Output both scale and bias per channel.
		self.fc = nn.Linear(cond_dim, channels * 2)

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (Tensor): Input tensor of shape (B, C, T).
			cond (Tensor): Conditioning tensor of shape (B, cond_dim).
		Returns:
			Tensor: Modulated tensor, of shape (B, C, T).
		"""
		# Compute scale and bias.
		gamma_beta = self.fc(cond)  # Shape: (B, C*2)
		gamma, beta = rearrange(gamma_beta, 'b (c n) -> n b c', n=2)  # Each shape: (B, C)
		gamma = gamma.unsqueeze(-1)  # (B, C, 1)
		beta = beta.unsqueeze(-1)    # (B, C, 1)
		# Apply FiLM modulation.
		return (1 + gamma) * x + beta

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
	A 1D U-Net for denoising a temporal action sequence using multiple residual blocks per stage.

	The network architecture is an encoder-decoder with skip connections:
	  - Initial convolution converts input sequence from (B, T, action_dim) to (B, action_dim, T).
	  - Encoder:
		  Stage 1: Two residual blocks -> skip connection -> downsampling.
		  Stage 2: Two residual blocks -> skip connection -> downsampling.
	  - Bottleneck: Two residual blocks.
	  - Decoder:
		  Stage 2: Upsampling -> concatenate skip connection from encoder stage 2 -> Two residual blocks.
		  Stage 1: Upsampling -> concatenate skip connection from encoder stage 1 -> Two residual blocks.
	  - Final convolution projects features back to the action dimension.

	Diagram:
	   Input (B, T, action_dim)
				  |
		 [Rearrange: (B, action_dim, T)]
				  |
			Initial Conv
				  |
	   +----------------------+
	   |   Encoder Stage 1    |  
	   |  [ResBlock x2]       |
	   |      Skip1           |
	   +----------------------+
				  |
			 Downsample1
				  |
	   +----------------------+
	   |   Encoder Stage 2    |
	   |  [ResBlock x2]       |
	   |      Skip2           |
	   +----------------------+
				  |
			 Downsample2
				  |
		 [Bottleneck: ResBlock x2]
				  |
			 Upsample2
				  |
		   Concatenate Skip2
				  |
		  [ResBlock x2 Decoder]
				  |
			 Upsample1
				  |
		   Concatenate Skip1
				  |
		  [ResBlock x2 Decoder]
				  |
			Final Convolution
				  |
		 Rearrange: (B, T, action_dim)
				  |
			   Output
				  
	Attributes:
		initial_conv (nn.Conv1d): Initial convolution layer.
		down_block1_1, down_block1_2 (ResidualBlock1D): Residual blocks for encoder stage 1.
		downsample1 (nn.Conv1d): Downsampling after stage 1.
		down_block2_1, down_block2_2 (ResidualBlock1D): Residual blocks for encoder stage 2.
		downsample2 (nn.Conv1d): Downsampling after stage 2.
		bottleneck_block1, bottleneck_block2 (ResidualBlock1D): Bottleneck residual blocks.
		up_block2_1, up_block2_2 (ResidualBlock1D): Residual blocks for decoder stage 2.
		upsample2 (nn.ConvTranspose1d): Upsampling for decoder stage 2.
		up_block1_1, up_block1_2 (ResidualBlock1D): Residual blocks for decoder stage 1.
		upsample1 (nn.ConvTranspose1d): Upsampling for decoder stage 1.
		final_conv (nn.Conv1d): Final convolution to project back to action dimension.
	"""
	def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int = 64):
		super().__init__()
		# Convert input (B, T, action_dim) to (B, action_dim, T)
		# Initial convolution.
		self.initial_conv = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
		
		# Encoder: Two stages, each with two residual blocks.
		self.down_block1_1 = ResidualBlock1D(hidden_dim, hidden_dim * 2, cond_dim)
		self.down_block1_2 = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim)
		self.downsample1 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1)

		self.down_block2_1 = ResidualBlock1D(hidden_dim * 2, hidden_dim * 4, cond_dim)
		self.down_block2_2 = ResidualBlock1D(hidden_dim * 4, hidden_dim * 4, cond_dim)
		self.downsample2 = nn.Conv1d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, stride=2, padding=1)
		
		# Bottleneck: two residual blocks.
		self.bottleneck_block1 = ResidualBlock1D(hidden_dim * 4, hidden_dim * 4, cond_dim)
		self.bottleneck_block2 = ResidualBlock1D(hidden_dim * 4, hidden_dim * 4, cond_dim)
		
		# Decoder: Two stages, each with two residual blocks and upsampling.
		self.up_block2_1 = ResidualBlock1D(hidden_dim * 4 + hidden_dim * 4, hidden_dim * 2, cond_dim)
		self.up_block2_2 = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, cond_dim)
		self.upsample2 = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
		
		self.up_block1_1 = ResidualBlock1D(hidden_dim * 2 + hidden_dim * 2, hidden_dim, cond_dim)
		self.up_block1_2 = ResidualBlock1D(hidden_dim, hidden_dim, cond_dim)
		self.upsample1 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
		
		# Final projection.
		self.final_conv = nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1)

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x (Tensor): Input tensor of shape (B, T, action_dim).
			cond (Tensor): Global conditioning tensor of shape (B, cond_dim).
		Returns:
			Tensor: Output tensor of shape (B, T, action_dim).
		"""
		# Rearrange to (B, action_dim, T).
		x = rearrange(x, 'b t a -> b a t')
		x0 = self.initial_conv(x)
		
		# Encoder stage 1.
		x1 = self.down_block1_1(x0, cond)
		x1 = self.down_block1_2(x1, cond)
		skip1 = x1  # Save for skip connection.
		x1_down = self.downsample1(x1)
		
		# Encoder stage 2.
		x2 = self.down_block2_1(x1_down, cond)
		x2 = self.down_block2_2(x2, cond)
		skip2 = x2  # Save for skip connection.
		x2_down = self.downsample2(x2)
		
		# Bottleneck.
		x_b = self.bottleneck_block1(x2_down, cond)
		x_b = self.bottleneck_block2(x_b, cond)
		
		# Decoder stage 2.
		# Upsample and concatenate skip from encoder stage 2.
		x_up2 = self.upsample2(x_b)
		# Ensure dimensions match for concatenation.
		x_up2 = torch.cat([x_up2, skip2], dim=1)
		x_up2 = self.up_block2_1(x_up2, cond)
		x_up2 = self.up_block2_2(x_up2, cond)
		
		# Decoder stage 1.
		x_up1 = self.upsample1(x_up2)
		x_up1 = torch.cat([x_up1, skip1], dim=1)
		x_up1 = self.up_block1_1(x_up1, cond)
		x_up1 = self.up_block1_2(x_up1, cond)
		
		out = self.final_conv(x_up1)
		# Rearrange back to (B, T, action_dim).
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
	def __init__(self, action_dim, condition_dim, time_embed_dim=128, window_size=WINDOW_SIZE+1):
		super(DiffusionPolicy, self).__init__()
		self.window_size = int(window_size)
		self.time_embed_dim = time_embed_dim

		# Dedicated time embedding module.
		self.time_embedding = DiffusionTimeEmbedding(time_embed_dim)

		# Visual encoder to extract features from images.
		self.visual_encoder = VisualEncoder()

		# Combined conditioning: two states (2 * 2) + two images (2 * IMAGE_FEATURE_DIM = 128).
		combined_dim = 2 * (2 + IMAGE_FEATURE_DIM)  # Expected 132.
		self.global_cond_dim = combined_dim + time_embed_dim  # 132 + 128 = 260.

		# Use the U-Net for denoising.
		self.unet = UNet1D(action_dim=int(action_dim), cond_dim=self.global_cond_dim, hidden_dim=64)

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

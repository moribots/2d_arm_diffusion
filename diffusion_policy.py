"""
This module implements the diffusion policy network used in the reverse diffusion process.
It predicts the noise added to an action sequence given conditioning information.
"""

import math
import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from config import *
from visual_encoder import VisualEncoder  # For extracting visual features from images

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

class FiLMBlock(nn.Module):
	"""
	Applies Feature-wise Linear Modulation (FiLM) to condition intermediate features.
	"""
	def __init__(self, channels, cond_dim):
		super().__init__()
		self.fc = nn.Linear(cond_dim, channels * 2)

	def forward(self, x, cond):
		# Compute scaling (gamma) and shifting (beta) parameters.
		gamma_beta = self.fc(cond)
		gamma, beta = gamma_beta.chunk(2, dim=-1)
		gamma = gamma.unsqueeze(-1)
		beta = beta.unsqueeze(-1)
		# Apply modulation with einsum.
		modulated = torch.einsum("bct, bcz -> bct", x, (1 + gamma))
		return modulated + beta

class DownBlock(nn.Module):
	"""
	Downsampling block for the U-Net encoder.
	Applies convolution, FiLM conditioning, and GELU activation.
	"""
	def __init__(self, in_channels, out_channels, cond_dim):
		super().__init__()
		# Ensure channel dimensions are integers.
		self.conv = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=3, padding=1)
		self.film = FiLMBlock(out_channels, cond_dim)
		self.act = nn.GELU()

	def forward(self, x, cond):
		x = self.conv(x)         # Convolution.
		x = self.film(x, cond)     # Apply FiLM conditioning.
		x = self.act(x)          # Activation.
		return x

class UpBlock(nn.Module):
	"""
	Upsampling block for the U-Net decoder.
	Concatenates skip connections, then applies convolution, FiLM conditioning, and GELU activation.
	"""
	def __init__(self, in_channels, out_channels, cond_dim):
		super().__init__()
		self.conv = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=3, padding=1)
		self.film = FiLMBlock(out_channels, cond_dim)
		self.act = nn.GELU()

	def forward(self, x, skip, cond):
		x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension.
		x = self.conv(x)
		x = self.film(x, cond)
		x = self.act(x)
		return x

class UNet1D(nn.Module):
	"""
	A 1D U-Net for denoising a temporal action sequence.
	Uses skip connections and FiLM conditioning.
	"""
	def __init__(self, action_dim, cond_dim, hidden_dim=64):
		super().__init__()
		# Input is (B, T, action_dim); use Conv1d so channels must be int.
		self.initial_conv = nn.Conv1d(int(action_dim), int(hidden_dim), kernel_size=3, padding=1)
		self.down1 = DownBlock(hidden_dim, hidden_dim * 2, cond_dim)
		self.down2 = DownBlock(hidden_dim * 2, hidden_dim * 4, cond_dim)
		self.bottleneck = nn.Conv1d(int(hidden_dim * 4), int(hidden_dim * 4), kernel_size=3, padding=1)
		self.up2 = UpBlock(int(hidden_dim * 4 + hidden_dim * 4), int(hidden_dim * 2), cond_dim)
		self.up1 = UpBlock(int(hidden_dim * 2 + hidden_dim * 2), int(hidden_dim), cond_dim)
		self.final_conv = nn.Conv1d(int(hidden_dim), int(action_dim), kernel_size=3, padding=1)

	def forward(self, x, cond):
		# Rearrange input from (B, T, action_dim) to (B, action_dim, T)
		x = rearrange(x, 'b t a -> b a t')
		x0 = self.initial_conv(x)  # Initial convolution.
		x1 = self.down1(x0, cond)  # Downsampling block.
		x2 = self.down2(x1, cond)  # Second downsampling.
		x_b = self.bottleneck(x2)  # Bottleneck.
		x_up2 = self.up2(x_b, x2, cond)  # First upsampling.
		x_up1 = self.up1(x_up2, x1, cond)  # Second upsampling.
		out = self.final_conv(x_up1)  # Final convolution.
		# Rearrange back to (B, T, action_dim).
		out = rearrange(out, 'b a t -> b t a')
		return out

class DiffusionPolicy(nn.Module):
	"""
	DiffusionPolicy predicts the noise added to an action sequence during the diffusion process.
	
	It conditions on:
	  - Agent state (4 numbers from t-1 and t).
	  - Visual features from two images (each produces 64 features, total 128).
	  - Diffusion timestep embedding (128 dimensions).
	  
	Global conditioning dimension: 4 + 128 + 128 = 260.
	"""
	def __init__(self, action_dim, condition_dim, time_embed_dim=128, window_size=WINDOW_SIZE+1):
		super(DiffusionPolicy, self).__init__()
		self.window_size = int(window_size)
		self.time_embed_dim = time_embed_dim

		# Time embedding network.
		self.time_mlp = nn.Sequential(
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU()
		)

		# Visual encoder to extract features from images.
		self.visual_encoder = VisualEncoder()

		# Combined conditioning: state (4) + two images (2 * IMAGE_FEATURE_DIM = 128).
		combined_dim = 4 + 2 * IMAGE_FEATURE_DIM  # Expected 132.
		self.global_cond_dim = combined_dim + time_embed_dim  # 132 + 128 = 260.

		# 1D U-Net for denoising.
		self.unet = UNet1D(action_dim=int(action_dim), cond_dim=self.global_cond_dim, hidden_dim=64)

	def forward(self, x, t, state, image):
		"""
		Forward pass of the DiffusionPolicy.
		
		Args:
			x (Tensor): Noised action sequence (B, window_size, action_dim).
			t (Tensor): Diffusion timestep (B,).
			state (Tensor): Agent state (B, 4).
			image (Tensor or list/tuple): Either a single image or a pair [img_t-1, img_t],
										  each of shape (B, 3, IMG_RES, IMG_RES).
										  
		Returns:
			Tensor: Predicted noise (B, window_size, action_dim).
		"""
		if x.dim() == 2:
			x = x.unsqueeze(1)
		# Process image input.
		if isinstance(image, (list, tuple)) and len(image) == 2:
			img_feat0 = self.visual_encoder(image[0])  # Features from image at t-1.
			img_feat1 = self.visual_encoder(image[1])  # Features from image at t.
			image_features = torch.cat([img_feat0, img_feat1], dim=-1)  # Shape: (B, 128)
		else:
			if image.dim() == 3:
				image = image.unsqueeze(0)
			feat = self.visual_encoder(image)
			image_features = torch.cat([feat, feat], dim=-1)
		# Process diffusion timestep.
		t = rearrange(t, 'b -> b 1').float()
		t_emb = get_sinusoidal_embedding(t, self.time_embed_dim)
		t_emb = self.time_mlp(t_emb)
		# Concatenate state and image features.
		combined_condition = torch.cat([state, image_features], dim=-1)  # Shape: (B, 132)
		global_cond = torch.cat([combined_condition, t_emb], dim=-1)  # Shape: (B, 260)
		noise_pred = self.unet(x, global_cond)
		return noise_pred

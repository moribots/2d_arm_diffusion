"""
Provides DiffusionPolicyInference, which wraps the diffusion policy model,
handles checkpoint loading, and implements DDIM sampling for inference.
"""

import torch
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from einops import rearrange
from config import *
from normalize import Normalize
import numpy as np
import os
import torch.nn.functional as F

class DiffusionPolicyInference:
	def __init__(self, model_path=OUTPUT_DIR + "diffusion_policy.pth", T=1000, device=None, norm_stats_path=OUTPUT_DIR + "normalization_stats.parquet"):
		"""
		Initialize DiffusionPolicyInference.

		Args:
			model_path (str): Path to the model checkpoint.
			T (int): Total diffusion timesteps.
			device: Torch device.
			norm_stats_path (str): Path to normalization stats (Parquet file).
		"""
		self.T = T
		self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
		# Instantiate the diffusion policy network, ensuring proper int conversion.
		self.model = DiffusionPolicy(
			action_dim=int(ACTION_DIM),
			condition_dim=int(CONDITION_DIM),
			time_embed_dim=128,
			window_size=int(WINDOW_SIZE + 1)
		).to(self.device)
		new_state_dict = {}
		if model_path and os.path.exists(model_path):
			checkpoint = torch.load(model_path, map_location=self.device)
			for key, value in checkpoint.items():
				new_state_dict[key.replace("module.", "")] = value
			self.model.load_state_dict(new_state_dict)
			print(f"Loaded checkpoint from {model_path}")
		self.model.eval()
		self.betas = get_beta_schedule(self.T).to(self.device)
		self.alphas, self.alphas_cumprod = compute_alphas(self.betas)
		self.normalize = Normalize.load(norm_stats_path, device=self.device)

	@torch.inference_mode()
	def sample_action(self, state, image, num_ddim_steps=200, smoothing=True):
		"""
		Generate a predicted action sequence using DDIM sampling.

		Args:
			state (Tensor): Conditioning state (1, 4) for t-1 and t.
			image (Tensor or list/tuple): Either a single image or a pair [img_t-1, img_t],
										  each of shape (B, 3, IMG_RES, IMG_RES).
			num_ddim_steps (int): Number of DDIM sampling steps (default: 200).
			smoothing (bool): Whether to apply temporal smoothing to the output.

		Returns:
			Tensor: Predicted action sequence of shape (window_size, ACTION_DIM).
		"""
		# Normalize the state.
		state = self.normalize.normalize_condition(state)
		eps = 1e-5
		max_clipped = 1.0
		
		# Create initial noise tensor with explicit integer dimensions.
		x_t = torch.randn((int(1), int(WINDOW_SIZE + 1), int(ACTION_DIM)), device=self.device)
		
		# Use more sophisticated timestep selection with cosine pacing
		timesteps = torch.linspace(self.T-1, 0, num_ddim_steps+1, device=self.device).round().long()
		
		# DDIM sampling with more steps for smoother denoising
		for i in range(len(timesteps) - 1):
			t_val = timesteps[i].item()
			t_next = timesteps[i+1].item()
			t_tensor = torch.tensor([t_val], device=self.device).float()
			alpha_bar_t = self.alphas_cumprod[t_val].view(1, 1, 1)
			
			# Predict noise using the diffusion policy.
			eps_pred = self.model(x_t, t_tensor, state, image)
			
			# Predict x0 with more stable numerical computation
			x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t + eps) * eps_pred) / torch.sqrt(alpha_bar_t + eps)
			x0_pred = torch.clamp(x0_pred, -max_clipped, max_clipped)
			
			# Calculate next timestep with DDIM formulation
			alpha_bar_t_next = self.alphas_cumprod[t_next].view(1, 1, 1) if t_next >= 0 else torch.tensor([1.0], device=self.device).view(1, 1, 1)
			
			# Improved DDIM update with less noise for later steps
			sigma_t = 0.0
			if t_next > 0:
				sigma_t = self.betas[t_next] * (1 - alpha_bar_t) / (1 - alpha_bar_t_next)
				sigma_t = torch.sqrt(sigma_t)
				
			# DDIM update with controlled noise
			noise = torch.randn_like(x_t) if t_next > 0 else 0
			x_t = torch.sqrt(alpha_bar_t_next) * x0_pred + \
				  torch.sqrt(1 - alpha_bar_t_next - sigma_t**2) * eps_pred + \
				  sigma_t * noise
			x_t = torch.clamp(x_t, -max_clipped, max_clipped)
		
		predicted_sequence_normalized = x_t[0, 1:, :]
		
		# Apply temporal smoothing if enabled
		if smoothing:
			kernel_size = 3
			 # Create a smoothed version of the actions
			smoothed_sequence = torch.zeros_like(predicted_sequence_normalized)
			
			# Apply smoothing separately for each action dimension
			for dim in range(predicted_sequence_normalized.shape[1]):
				# Extract this dimension and add batch and channel dimensions
				dim_data = predicted_sequence_normalized[:, dim].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
				
				# Create 1D kernel for convolution
				kernel = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
				
				# Pad to maintain sequence length
				padded = F.pad(dim_data, (kernel_size//2, kernel_size//2), mode='replicate')
				
				# Apply convolution for smoothing
				smoothed = F.conv1d(padded, kernel)
				
				# Store the smoothed result
				smoothed_sequence[:, dim] = smoothed[0, 0, :]
			
			predicted_sequence_normalized = smoothed_sequence
		
		predicted_sequence = self.normalize.unnormalize_action(predicted_sequence_normalized)
		
		# Check if any values were clamped
		before_clamp = predicted_sequence.clone()
		predicted_sequence = torch.clamp(
			predicted_sequence,
			min=torch.tensor([0, 0], device=self.device),
			max=torch.tensor([SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1], device=self.device))
		
		if not torch.equal(before_clamp, predicted_sequence):
			print(f'Warning: Some predicted actions were clamped to be within screen bounds {before_clamp[0]}')
			print(f'Normalized predicted sequence: {predicted_sequence_normalized}')
		
		return predicted_sequence

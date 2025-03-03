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
		else:
			print(f"No checkpoint found at {model_path}, starting from scratch.")
		self.model.eval()
		self.betas = get_beta_schedule(self.T).to(self.device)
		self.alphas, self.alphas_cumprod = compute_alphas(self.betas)
		self.normalize = Normalize.load(norm_stats_path, device=self.device)

	@torch.no_grad()
	def sample_action(self, state, image, num_ddim_steps=100):
		"""
		Generate a predicted action sequence using DDIM sampling.

		Args:
			state (Tensor): Conditioning state (1, 4) for t-1 and t.
			image (Tensor or list/tuple): Either a single image or a pair [img_t-1, img_t],
										  each of shape (B, 3, IMG_RES, IMG_RES).
			num_ddim_steps (int): Number of DDIM sampling steps (default: 100).

		Returns:
			Tensor: Predicted action sequence of shape (window_size, ACTION_DIM).
		"""
		# Normalize the state.
		state = self.normalize.normalize_condition(state)
		eps = 1e-5
		max_clipped = 1.0
		# Create initial noise tensor with explicit integer dimensions.
		x_t = torch.randn((int(1), int(WINDOW_SIZE + 1), int(ACTION_DIM)), device=self.device)
		ddim_timesteps = np.linspace(0, self.T - 1, num_ddim_steps, dtype=int)
		ddim_timesteps = list(ddim_timesteps)[::-1]
		for i in range(len(ddim_timesteps) - 1):
			t_val = ddim_timesteps[i]
			t_next = ddim_timesteps[i + 1]
			t_tensor = torch.tensor([t_val], device=self.device).float()
			alpha_bar_t = self.alphas_cumprod[t_val].view(1, 1, 1)
			# Predict noise using the diffusion policy.
			eps_pred = self.model(x_t, t_tensor, state, image)
			x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t + eps)
			x0_pred = torch.clamp(x0_pred, -max_clipped, max_clipped)
			alpha_bar_t_next = self.alphas_cumprod[t_next].view(1, 1, 1)
			x_t = torch.sqrt(alpha_bar_t_next) * x0_pred + torch.sqrt(1 - alpha_bar_t_next) * eps_pred
			x_t = torch.clamp(x_t, -max_clipped, max_clipped)
		predicted_sequence_normalized = x_t[0, 1:, :]
		predicted_sequence = self.normalize.unnormalize_action(predicted_sequence_normalized)
		# Check if any values were clamped
		before_clamp = predicted_sequence.clone()
		predicted_sequence = torch.clamp(
			predicted_sequence,
			min=torch.tensor([0, 0], device=self.device),
			max=torch.tensor([SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1], device=self.device)
		)
		if not torch.equal(before_clamp, predicted_sequence):
			print("Warning: Some predicted actions were clamped to be within screen bounds")
		return predicted_sequence

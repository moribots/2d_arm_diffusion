"""
Provides DiffusionPolicyInference, which wraps the diffusion policy model,
handles checkpoint loading, and implements DDIM sampling for inference.
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from src.config import *
from src.diffusion.diffusion_policy import DiffusionPolicy
from src.utils.diffusion_utils import get_beta_schedule, compute_alphas
from src.utils.normalize import Normalize
from src.seed import set_seed
set_seed(42)

class DiffusionPolicyInference:
	"""
	Inference wrapper for the diffusion policy model.

	This class implements DDIM sampling for generating a full action sequence,
	and maintains an action buffer for temporal continuity.

	Attributes:
		T (int): Total diffusion timesteps.
		device (torch.device): Device for inference.
		model (DiffusionPolicy): The diffusion policy model.
		betas (Tensor): Beta schedule tensor.
		alphas (Tensor): Alpha values computed from betas.
		alphas_cumprod (Tensor): Cumulative product of alphas.
		normalize (Normalize): Normalization utility for unnormalizing actions.
		action_buffer (list): Buffer to store partial action sequences for continuity.
		current_action_idx (int): Index for retrieving the next action from the buffer.
		last_full_sequence (Tensor): The last full generated action sequence (for warm starting).
	"""
	def __init__(self, model_path, T=1000, device=None, norm_stats_path=None):
		# Use default normalization stats path if none provided.
		if norm_stats_path is None:
			norm_stats_path = os.path.join(OUTPUT_DIR, DATA_SOURCE_DIR, DATASET_TYPE, NORM_STATS_FILENAME)
			
		self.T = T
		self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
		
		# Instantiate the diffusion policy network with proper dimensions.
		self.model = DiffusionPolicy(
			action_dim=int(ACTION_DIM),
			condition_dim=int(CONDITION_DIM),
			window_size=int(WINDOW_SIZE)
		).to(self.device)
		
		# Load model checkpoint (assumed to contain EMA-updated parameters).
		if model_path and os.path.exists(model_path):
			checkpoint = torch.load(model_path, map_location=self.device)
			# Remove DataParallel prefix if present.
			new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
			self.model.load_state_dict(new_state_dict)
			print(f"Loaded checkpoint from {model_path}")
		else:
			print("Warning: No valid model checkpoint provided.")
		
		self.model.eval()
		
		# Prepare diffusion schedule.
		self.betas = get_beta_schedule(self.T).to(self.device)
		self.alphas, self.alphas_cumprod = compute_alphas(self.betas)
		
		# Load normalization statistics for unnormalizing actions.
		self.normalize = Normalize.load(norm_stats_path, device=self.device)

		# Initialize the action buffer and pointer for warm-started sampling.
		self.action_buffer = []
		self.current_action_idx = 0
		self.last_full_sequence = None  # For maintaining temporal continuity

	def generate_action_sequence(self, state, image, num_ddim_steps=20):
		"""
		Generate a full action sequence using DDIM sampling.

		Args:
			state (Tensor): Normalized state tensor (shape: [1, feature_dim]).
			image (Tensor or list): Conditioning image(s).
			num_ddim_steps (int): Number of DDIM sampling steps.

		Returns:
			Tensor: Predicted action sequence of shape ([WINDOW_SIZE], ACTION_DIM).
		"""
		max_clipped = 1.0

		# Initialize the diffusion process with random noise.
		x_t = torch.randn((1, WINDOW_SIZE, ACTION_DIM), device=self.device)

		# Select timesteps for DDIM sampling using a cosine-paced schedule.
		timesteps = torch.linspace(self.T - 1, 0, num_ddim_steps + 1, device=self.device).round().long()

		# Iteratively refine the noise estimate.
		for i in range(len(timesteps) - 1):
			t_val = timesteps[i].item()
			t_next = timesteps[i + 1].item()
			t_tensor = torch.tensor([t_val], device=self.device).float()
			alpha_bar_t = self.alphas_cumprod[t_val].view(1, 1, 1)

			# Predict noise using the diffusion policy model.
			eps_pred = self.model(x_t, t_tensor, state, image)

			# Estimate the original (clean) action sequence.
			x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
			x0_pred = torch.clamp(x0_pred, -max_clipped, max_clipped)
			alpha_bar_t_next = self.alphas_cumprod[t_next].view(1, 1, 1)
			# Update x_t to the next diffusion step.
			x_t = torch.sqrt(alpha_bar_t_next) * x0_pred + torch.sqrt(1 - alpha_bar_t_next) * eps_pred
			x_t = torch.clamp(x_t, -max_clipped, max_clipped)

		# Remove the first timestep (warm-up) and retrieve the predicted sequence.
		predicted_sequence_normalized = x_t[0, 1:, :]

		# Unnormalize the predicted action sequence.
		predicted_sequence = self.normalize.unnormalize_action(predicted_sequence_normalized)

		# Clamp actions within valid screen bounds.
		before_clamp = predicted_sequence.clone()
		predicted_sequence = torch.clamp(
			predicted_sequence,
			min=torch.tensor([0, 0], device=self.device),
			max=torch.tensor([SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1], device=self.device)
		)
		if not torch.equal(before_clamp, predicted_sequence):
			print(f'Warning: Some predicted actions were clamped from {before_clamp[0]} to {predicted_sequence[0]}')
			print(f'Normalized predicted sequence: {predicted_sequence_normalized}')

		return predicted_sequence  # Shape: (WINDOW_SIZE, ACTION_DIM)

	@torch.inference_mode()
	def sample_action(self, state, image, num_ddim_steps=20):
		"""
		Generate a predicted action (or interpolated action) based on the current buffer state.

		If the action buffer is exhausted, a new full sequence is generated.
		The new sequence can be warm-started using the tail of the previous full sequence
		to ensure temporal continuity.

		Args:
			state (Tensor): Raw state tensor.
			image (Tensor or list): Conditioning image(s).
			num_ddim_steps (int): Number of DDIM sampling steps.

		Returns:
			tuple: (action_tensor, is_new_inference) where is_new_inference is True if a new sequence was generated.
		"""
		# Normalize the state for proper conditioning.
		normalized_state = self.normalize.normalize_condition(state)

		is_new_inference = False

		# Generate a new sequence if the buffer is empty or exhausted.
		if len(self.action_buffer) == 0 or self.current_action_idx >= len(self.action_buffer):
			is_new_inference = True
			full_sequence = self.generate_action_sequence(normalized_state, image, num_ddim_steps)
			# For execution, only a portion of the sequence is used.
			buffer_size = min(WINDOW_SIZE // 2, full_sequence.shape[0])
			self.action_buffer = full_sequence[:buffer_size]
			self.current_action_idx = 0

		# Retrieve the next action from the buffer.
		action = self.action_buffer[self.current_action_idx]
		self.current_action_idx += 1

		return action, is_new_inference

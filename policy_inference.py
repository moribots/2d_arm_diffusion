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
import time
# Set random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)

class DiffusionPolicyInference:
	"""
	Inference wrapper for the diffusion policy.

	Generates a full action sequence via DDIM sampling and maintains an action buffer.
	The buffer is re-filled using a warm start from the previous full sequence to ensure
	temporal continuity.
	"""
	def __init__(self, model_path, T=1000, device=None, norm_stats_path="normalization_stats.parquet"):
		# (Initialization code remains unchanged)
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

		# Initialize action buffer, index, and last action sequence for warm-starting.
		self.action_buffer = []
		self.current_action_idx = 0
		self.last_full_sequence = None  # Stores the full previously generated action sequence

	def _generate_action_sequence(self, state, image, num_ddim_steps=100, smoothing=True, warm_start=None):
		"""
		Generate a full action sequence using DDIM sampling.

		If a warm_start is provided (as a sequence with shape [warm_len, ACTION_DIM]),
		the first warm_len timesteps of the new sequence are initialized from it.
		The remainder is filled with small noise to help the diffusion process refine the sequence.

		Args:
			state (Tensor): Normalized state tensor (shape: [1, feature_dim]).
			image (Tensor or list): Conditioning image(s).
			num_ddim_steps (int): Number of DDIM sampling steps.
			smoothing (bool): If True, apply temporal smoothing to the output.
			warm_start (Tensor, optional): A warm start sequence (shape: [warm_len, ACTION_DIM]).

		Returns:
			Tensor: Predicted action sequence of shape ([WINDOW_SIZE], ACTION_DIM).
		"""
		max_clipped = 1.0

		# Initialize the diffusion process.
		if False: # warm_start is not None:
			# Use the provided warm_start sequence to initialize the beginning timesteps.
			warm_len = warm_start.shape[0]
			x_t = torch.zeros((1, WINDOW_SIZE + 1, ACTION_DIM), device=self.device)
			x_t[0, :warm_len, :] = warm_start  # fill first warm_len steps with warm_start
			# Fill remaining timesteps with small random noise.
			noise = 0.0001 * torch.randn((1, WINDOW_SIZE + 1 - warm_len, ACTION_DIM), device=self.device)
			x_t[0, warm_len:, :] = noise
		else:
			print(f'No warm start provided; initializing full sequence from standard Gaussian noise.')
			x_t = torch.randn((1, WINDOW_SIZE + 1, ACTION_DIM), device=self.device)

		# Select timesteps for DDIM sampling (cosine-paced).
		timesteps = torch.linspace(self.T - 1, 0, num_ddim_steps + 1, device=self.device).round().long()

		# Perform DDIM sampling iteratively.
		for i in range(len(timesteps) - 1):
			t_val = timesteps[i].item()
			t_next = timesteps[i + 1].item()
			t_tensor = torch.tensor([t_val], device=self.device).float()
			alpha_bar_t = self.alphas_cumprod[t_val].view(1, 1, 1)

			# Predict noise using the diffusion policy.
			eps_pred = self.model(x_t, t_tensor, state, image)

			# Compute x0 prediction.
			x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
			x0_pred = torch.clamp(x0_pred, -max_clipped, max_clipped)
			alpha_bar_t_next = self.alphas_cumprod[t_next].view(1, 1, 1)
			# Update x_t to be the next step in the sequence.
			x_t = torch.sqrt(alpha_bar_t_next) * x0_pred + torch.sqrt(1 - alpha_bar_t_next) * eps_pred
			x_t = torch.clamp(x_t, -max_clipped, max_clipped)

		# Remove the first element which corresponds to the t-1 action.
		predicted_sequence_normalized = x_t[0, 1:, :]

		# Optionally apply temporal smoothing.
		if smoothing:
			kernel_size = 1  # adjust kernel size to control smoothing level
			smoothed_sequence = torch.zeros_like(predicted_sequence_normalized)
			for dim in range(predicted_sequence_normalized.shape[1]):
				dim_data = predicted_sequence_normalized[:, dim].unsqueeze(0).unsqueeze(0)  # shape: [1, 1, seq_len]
				kernel = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
				padded = F.pad(dim_data, (kernel_size // 2, kernel_size // 2), mode='replicate')
				smoothed = F.conv1d(padded, kernel)
				smoothed_sequence[:, dim] = smoothed[0, 0, :]
			predicted_sequence_normalized = smoothed_sequence

		# Unnormalize the action sequence.
		predicted_sequence = self.normalize.unnormalize_action(predicted_sequence_normalized)

		# Clamp actions to be within screen bounds.
		before_clamp = predicted_sequence.clone()
		predicted_sequence = torch.clamp(
			predicted_sequence,
			min=torch.tensor([0, 0], device=self.device),
			max=torch.tensor([SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1], device=self.device)
		)
		if not torch.equal(before_clamp, predicted_sequence):
			print(f'Warning: Some predicted actions were clamped from {before_clamp[0]} to {predicted_sequence[0]}')
			print(f'Normalized predicted sequence: {predicted_sequence_normalized}')

		return predicted_sequence  # shape: (WINDOW_SIZE, ACTION_DIM)

	@torch.inference_mode()
	def sample_action(self, state, image, num_ddim_steps=200, smoothing=False):
		"""
		Generate a predicted action or interpolated action based on the buffer state.

		If the action buffer is exhausted, a new sequence is generated.
		This new sequence is warm-started using the tail of the previous full sequence (if available)
		to maintain temporal continuity.

		Args:
			state (Tensor): Raw state tensor.
			image (Tensor or list): Image or images for conditioning.
			num_ddim_steps (int): Number of DDIM sampling steps.
			smoothing (bool): Whether to apply temporal smoothing.

		Returns:
			tuple: (action_tensor, is_new_inference) where is_new_inference indicates
				   whether a new sequence was generated.
		"""
		# Normalize state.
		normalized_state = self.normalize.normalize_condition(state)

		is_new_inference = False

		# Check if we need to generate a new action sequence.
		if len(self.action_buffer) == 0 or self.current_action_idx >= len(self.action_buffer):
			is_new_inference = True
			# Use the tail of the previous full sequence as warm start if available.
			warm_start = None
			if self.last_full_sequence is not None:
				# Use the first half-window of the previous sequence.
				warm_start = self.last_full_sequence[:(WINDOW_SIZE // 2)]
			# Generate a new full action sequence.
			full_sequence = self._generate_action_sequence(
				normalized_state, image, num_ddim_steps, smoothing, warm_start=warm_start
			)
			# Store the full sequence for future warm-starting.
			self.last_full_sequence = full_sequence.clone()
			# For execution, use only a portion of the sequence.
			buffer_size = min(WINDOW_SIZE // 2, full_sequence.shape[0])
			self.action_buffer = full_sequence[:buffer_size]
			self.current_action_idx = 0

		# Retrieve the next action from the buffer.
		action = self.action_buffer[self.current_action_idx]
		self.current_action_idx += 1

		return action, is_new_inference

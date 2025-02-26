import torch
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from einops import rearrange
from config import *
from normalize import Normalize  # New Normalize class
import numpy as np

class DiffusionPolicyInference:
	def __init__(self, model_path=OUTPUT_DIR + "diffusion_policy.pth", T=1000, device=None, norm_stats_path=OUTPUT_DIR + "normalization_stats.json"):
		self.T = T
		self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
		# Instantiate the model with the updated window size.
		self.model = DiffusionPolicy(action_dim=ACTION_DIM, condition_dim=CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE+1).to(self.device)
		# Load model trained from parallel gpus.
		new_state_dict = {}
		checkpoint = torch.load(model_path, map_location=self.device)
		for key, value in checkpoint.items():
			new_key = key.replace("module.", "")  # Remove "module." prefix
			new_state_dict[new_key] = value
		self.model.load_state_dict(new_state_dict)
		self.model.eval()
		self.betas = get_beta_schedule(self.T).to(self.device)
		self.alphas, self.alphas_cumprod = compute_alphas(self.betas)
		self.alphas = self.alphas.to(self.device)
		self.alphas_cumprod = self.alphas_cumprod.to(self.device)

		# Load normalization statistics using the Normalize class.
		self.normalize = Normalize.load(norm_stats_path, device=self.device)

	@torch.no_grad()
	def sample_action(self, state, image, num_ddim_steps=50):
		"""
		Generate a sample EE action using DDIM sampling (Denoising Diffusion Implicit Models) with reduced steps.
		
		DDIM is a deterministic sampling method derived from diffusion models that allows for much faster inference.
		Instead of iterating over all T diffusion timesteps (e.g., 1000), DDIM uses a reduced set of timesteps 
		(num_ddim_steps, e.g., 50) to approximate the reverse diffusion process.
		
		The key steps in DDIM are:
		1. From the current noisy sample x_t and predicted noise (epsilon) from the model, predict the original signal:
			x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
			where alpha_bar_t is the cumulative product of alphas at time t (a measure of the noise level).
		2. Update the sample using:
			x_t_next = sqrt(alpha_bar_t_next) * x0_pred + sqrt(1 - alpha_bar_t_next) * epsilon
			where alpha_bar_t_next corresponds to the next timestep.
		
		For numerical stability:
		- A small constant (eps) is added to denominators.
		- Intermediate x0_pred and x_t values are clamped to a fixed range.
		
		Parameters:
		state: Conditioning state tensor of shape (1, 4) representing the EE positions at t-1 and t.
		image: Conditioning image tensor of shape (1, 3, IMG_RES, IMG_RES).
		num_ddim_steps: The number of DDIM steps to use for sampling (default is 50).

		Returns:
		A tensor of shape (window_size, action_dim) representing the denoised action sequence.
		"""
		# Normalize the state condition (consistent with training)
		state = self.normalize.normalize_condition(state)
		
		eps = 1e-5  # Small constant for numerical stability
		# Initialize x_t as random noise.
		max_clipped = 3.0
		x_t = torch.randn((1, WINDOW_SIZE+1, ACTION_DIM), device=self.device)
		# Clip to the same range used in the DDIM sampling
		x_t = torch.clamp(x_t, -max_clipped, max_clipped)
		
		# Create a reduced set of timesteps for DDIM sampling (from high noise to low noise)
		ddim_timesteps = np.linspace(0, self.T - 1, num_ddim_steps, dtype=int)
		ddim_timesteps = list(ddim_timesteps)[::-1]  # e.g., from T-1 down to 0
		
		# For deterministic sampling, set eta = 0 (no additional noise is added)
		eta = 0.0

		# DDIM sampling loop over the reduced timesteps.
		for i in range(len(ddim_timesteps) - 1):
			t = ddim_timesteps[i]
			t_next = ddim_timesteps[i+1]
			t_tensor = torch.tensor([t], device=self.device).float()  # current timestep as tensor
			
			# Get the cumulative product of alphas for current timestep.
			alpha_bar_t = self.alphas_cumprod[t].view(1, 1, 1)
			
			# Predict noise epsilon at current timestep using the diffusion model.
			eps_pred = self.model(x_t, t_tensor, state, image)
			
			# Predict the original signal x0 from the noisy sample x_t and the predicted noise.
			x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t + eps)
			# Clamp x0_pred to a fixed range to avoid extreme values.
			x0_pred = torch.clamp(x0_pred, -max_clipped, max_clipped)
			
			# Get the cumulative product of alphas for the next timestep.
			alpha_bar_t_next = self.alphas_cumprod[t_next].view(1, 1, 1)
			
			# DDIM update rule: update x_t using the predicted x0 and epsilon.
			x_t = torch.sqrt(alpha_bar_t_next) * x0_pred + torch.sqrt(1 - alpha_bar_t_next) * eps_pred
			# Clamp x_t to prevent numerical explosion.
			x_t = torch.clamp(x_t, -max_clipped, max_clipped)
		
		# Return the full sequence except for the t-1th action.
		print("Raw x_t shape:", x_t.shape)
		# After the full denoising process is complete
		predicted_sequence_normalized = x_t[0, 1:, :]  # Your current code

		# Unnormalize to get actions in environment space
		predicted_sequence = self.normalize.unnormalize_action(predicted_sequence_normalized)

		# Clip the unnormalized actions to be within environment bounds
		predicted_sequence = torch.clamp(
			predicted_sequence,
			min=torch.tensor([0, 0], device=self.device),
			max=torch.tensor([SCREEN_WIDTH-1, SCREEN_HEIGHT-1], device=self.device)
		)
		return predicted_sequence

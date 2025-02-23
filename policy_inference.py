import torch
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from einops import rearrange
from config import ACTION_DIM, CONDITION_DIM, WINDOW_SIZE
from normalize import Normalize  # New Normalize class

class DiffusionPolicyInference:
	def __init__(self, model_path="diffusion_policy.pth", T=1000, device=None, norm_stats_path="normalization_stats.json"):
		self.T = T
		self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
		# Instantiate the model with the updated window size.
		self.model = DiffusionPolicy(action_dim=ACTION_DIM, condition_dim=CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE+1).to(self.device)
		self.model.load_state_dict(torch.load(model_path, map_location=self.device))
		self.model.eval()
		self.betas = get_beta_schedule(self.T).to(self.device)
		self.alphas, self.alphas_cumprod = compute_alphas(self.betas)
		self.alphas = self.alphas.to(self.device)
		self.alphas_cumprod = self.alphas_cumprod.to(self.device)

		# Load normalization statistics using the Normalize class.
		self.normalize = Normalize.load(norm_stats_path, device=self.device)

	@torch.no_grad()
	def sample_action(self, state, image):
		"""
		Generate a sample EE action given a conditioning state and image using the reverse diffusion process.
		After the reverse diffusion process, the predicted action (still in the normalized space)
		is un-normalized using the stored normalization statistics.
		"""
		eps = 1e-8  # Small constant to avoid division by zero
		x_t = torch.randn((1, WINDOW_SIZE+1, ACTION_DIM), device=self.device)
		for t in reversed(range(1, self.T)):
			t_tensor = torch.tensor([t], device=self.device).float()  # shape: (1,)
			alpha_t = rearrange(self.alphas[t:t+1], 'b -> b 1 1')
			alpha_bar_t = rearrange(self.alphas_cumprod[t:t+1], 'b -> b 1 1')
			beta_t = rearrange(self.betas[t:t+1], 'b -> b 1 1')
			eps_pred = self.model(x_t, t_tensor, state, image)
			coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + eps)
			# Use the epsilon in the denominator to avoid division by zero
			x_prev = (x_t - coef * eps_pred) / torch.sqrt(alpha_t + eps)
			if t > 1:
				z = torch.randn_like(x_t)
				x_prev = x_prev + torch.sqrt(beta_t + eps) * z
			x_t = x_prev
		pred_normalized = x_t[0, -1, :]  # (ACTION_DIM,)
		pred = self.normalize.unnormalize_action(pred_normalized)
		return pred


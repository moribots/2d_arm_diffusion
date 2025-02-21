import torch
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from einops import rearrange

class DiffusionPolicyInference:
	def __init__(self, model_path="diffusion_policy.pth", T=1000, device=None):
		self.T = T
		self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
		# Use condition_dim=3 so that the input dimension is 2 + 3 + 128 = 133.
		self.model = DiffusionPolicy(action_dim=2, condition_dim=3).to(self.device)
		self.model.load_state_dict(torch.load(model_path, map_location=self.device))
		self.model.eval()
		self.betas = get_beta_schedule(self.T).to(self.device)
		self.alphas, self.alphas_cumprod = compute_alphas(self.betas)
		self.alphas = self.alphas.to(self.device)
		self.alphas_cumprod = self.alphas_cumprod.to(self.device)
		
	@torch.no_grad()
	def sample_action(self, condition):
		"""
		Generate a sample EE action given a conditioning signal using the reverse diffusion process.
		The reverse (denoising) process follows:
		
		$$ x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}} \\left( x_t - \\frac{1-\\alpha_t}{\\sqrt{1-\\bar{\\alpha}_t}} \\epsilon_\\theta(x_t,t,c) \\right) + \\sqrt{\\beta_t} z $$
		
		where \\( z \\sim \\mathcal{N}(0,I) \\) (for \\(t > 1\\)).
		"""
		# Start with pure Gaussian noise.
		x_t = torch.randn((1, 2), device=self.device)
		# Reverse diffusion loop.
		for t in reversed(range(1, self.T)):
			t_tensor = torch.tensor([t], device=self.device).float()  # shape: (1,)
			alpha_t = self.alphas[t]
			alpha_bar_t = self.alphas_cumprod[t]
			beta_t = self.betas[t]
			# Predict the noise using the trained model.
			eps_pred = self.model(x_t, t_tensor, condition)
			# Compute coefficient: (1-α_t) / sqrt(1-ᾱ_t)
			coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
			# Compute x_{t-1} using the DDPM reverse formula.
			x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - coef * eps_pred)
			if t > 1:
				z = torch.randn_like(x_t)
				x_prev = x_prev + torch.sqrt(beta_t) * z
			x_t = x_prev
		return rearrange(x_t, '1 d -> d')  # Return tensor of shape (2,)

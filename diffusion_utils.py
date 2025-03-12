"""
Utility functions for the diffusion process.
"""

import torch

def get_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
	"""
	Generate a beta schedule using a squared cosine schedule as described in
	"Improved Denoising Diffusion Probabilistic Models".

	The schedule computes the cumulative product alpha_bar(t) using:
		alpha_bar(t) = cos(((t/T + s) / (1+s)) * (pi/2))^2,
	and then defines beta_t = 1 - alpha_bar(t+1) / alpha_bar(t) for t = 0,...,T-1.

	Args:
		T (int): Total number of diffusion timesteps.
		s (float): Small offset for stability (default: 0.008).

	Returns:
		torch.Tensor: Beta schedule of shape (T,). Each beta is clamped to at most 0.999.
	"""
	# Generate T+1 linearly spaced time points from 0 to T (inclusive)
	timesteps = torch.linspace(0, T, T + 1)
	# Compute cumulative product of alphas using the squared cosine function
	alpha_bar = torch.cos(((timesteps / (T + s)) / (1 + s)) * (torch.pi / 2)) ** 2
	betas = []
	for t in range(T):
		# Compute beta_t as the relative drop between consecutive alpha_bar values
		beta = 1 - (alpha_bar[t + 1] / alpha_bar[t])
		# Clamp beta to prevent numerical issues
		beta = min(beta.item(), 0.999)
		betas.append(beta)
	return torch.tensor(betas, dtype=torch.float32)

def compute_alphas(betas):
	"""
	Compute alpha values (1 - beta) and their cumulative product.
	
	Args:
		betas (Tensor): Tensor of beta values.
	
	Returns:
		Tuple[Tensor, Tensor]: (alphas, alphas_cumprod)
	"""
	alphas = 1.0 - betas
	alphas_cumprod = torch.cumprod(alphas, dim=0)
	return alphas, alphas_cumprod

"""
Utility functions for the diffusion process.
"""

import torch

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
	"""
	Generate a linear schedule of beta values for diffusion.
	
	Args:
		T (int): Number of timesteps.
		beta_start (float): Starting beta value.
		beta_end (float): Ending beta value.
	
	Returns:
		Tensor: Beta schedule of shape (T,).
	"""
	return torch.linspace(beta_start, beta_end, T)

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

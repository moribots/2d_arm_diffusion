import torch

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
	"""
	Returns a linear beta schedule for T timesteps.
	"""
	return torch.linspace(beta_start, beta_end, T)

def compute_alphas(betas):
	"""
	Computes alpha and the cumulative product \bar{alpha} from betas.
	"""
	alphas = 1.0 - betas
	alphas_cumprod = torch.cumprod(alphas, dim=0)
	return alphas, alphas_cumprod

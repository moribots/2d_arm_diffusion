"""
Module to set the random seed for reproducibility.
"""

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
	"""
	Set the random seed for reproducibility across Python, NumPy, and PyTorch.

	Args:
		seed (int): Seed value to use.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

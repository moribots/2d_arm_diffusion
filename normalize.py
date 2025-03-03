import torch
import pyarrow as pa
import pyarrow.parquet as pq
import os
import pandas as pd
from tabulate import tabulate
import numpy as np

class Normalize:
	"""
	Normalization module for computing, saving, and loading normalization statistics.
	
	For condition features, normalization is performed using mean and standard deviation.
	For actions, min–max normalization is applied.
	
	The computed statistics are registered as buffers to ensure consistency between training 
	and inference.
	"""
	def __init__(self, condition_mean: torch.Tensor, condition_std: torch.Tensor, 
				 action_min: torch.Tensor, action_max: torch.Tensor):
		"""
		Initialize normalization statistics and register them as buffers.

		Args:
			condition_mean (Tensor): Mean vector for condition features.
			condition_std (Tensor): Standard deviation for condition features.
			action_min (Tensor): Minimum vector for actions.
			action_max (Tensor): Maximum vector for actions.
		"""
		self.register_buffer('condition_mean', condition_mean)
		self.register_buffer('condition_std', condition_std)
		self.register_buffer('action_min', action_min)
		self.register_buffer('action_max', action_max)
	
	def register_buffer(self, name: str, tensor: torch.Tensor):
		"""
		Register a tensor as a buffer. In a full nn.Module, this would use register_buffer.
		Here, we simulate that by setting an attribute.
		"""
		setattr(self, name, tensor)
	
	@classmethod
	def compute_from_samples(cls, samples: list) -> "Normalize":
		"""
		Compute normalization statistics from a list of samples.

		Args:
			samples (list): List of sample dictionaries.

		Returns:
			Normalize: Instance with computed statistics.
		"""
		conditions = []
		actions_list = []
		for sample in samples:
			# Extract state for conditioning normalization.
			if "observation" in sample and "state" in sample["observation"]:
				state = torch.tensor(sample["observation"]["state"], dtype=torch.float32)
				cond = state.flatten()  # Expecting shape (4,)
				conditions.append(cond)
			elif "observation.state" in sample:
				state = torch.tensor(sample["observation.state"], dtype=torch.float32)
				cond = state.flatten()
				conditions.append(cond)
			else:
				raise KeyError("Sample must contain observation['state']")
			# Extract action data for action normalization.
			if "action" not in sample:
				raise KeyError("Sample does not contain 'action'")
			action_data = sample["action"]
			if isinstance(action_data[0], (list, tuple)):
				act = torch.tensor(action_data, dtype=torch.float32)
			else:
				act = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
			actions_list.append(act)
		conditions_cat = torch.stack(conditions, dim=0)
		condition_mean = conditions_cat.mean(dim=0)
		condition_std = conditions_cat.std(dim=0, unbiased=False) + 1e-6  # Avoid division by zero.
		actions_cat = torch.cat(actions_list, dim=0)
		action_min = actions_cat.min(dim=0)[0]
		action_max = actions_cat.max(dim=0)[0]
		return cls(condition_mean, condition_std, action_min, action_max)
	
	def normalize_condition(self, condition: torch.Tensor) -> torch.Tensor:
		"""
		Normalize a condition tensor using computed mean and std.

		Args:
			condition (Tensor): Condition tensor (shape: (4,) or (B, 4)).

		Returns:
			Tensor: Normalized condition.
		"""
		if condition.dim() == 1:
			cond_dim = condition.shape[0]
		else:
			cond_dim = condition.shape[-1]
		
		# Adjust mean and std for dimension mismatch
		if self.condition_mean.numel() != cond_dim:
			condition_mean = torch.cat([self.condition_mean, self.condition_mean], dim=0).unsqueeze(0)
			condition_std = torch.cat([self.condition_std, self.condition_std], dim=0).unsqueeze(0)
		else:
			condition_mean = self.condition_mean.unsqueeze(0) if condition.dim() > 1 else self.condition_mean
			condition_std = self.condition_std.unsqueeze(0) if condition.dim() > 1 else self.condition_std
		
		return (condition - condition_mean) / condition_std
	
	def normalize_action(self, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
		"""
		Normalize an action tensor using min–max normalization.

		Args:
			action (Tensor): Raw action tensor.
			eps (float): Epsilon to avoid division by zero.

		Returns:
			Tensor: Normalized action.
		"""
		return (action - self.action_min) / (self.action_max - self.action_min + eps)
	
	def unnormalize_action(self, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
		"""
		Revert normalization on an action tensor.

		Args:
			action (Tensor): Normalized action tensor.
			eps (float): Epsilon to avoid division by zero.

		Returns:
			Tensor: Original action tensor.
		"""
		return action * (self.action_max - self.action_min + eps) + self.action_min
	
	def to_dict(self) -> dict:
		"""
		Convert normalization statistics to a dictionary for saving.

		Returns:
			dict: Dictionary with keys 'condition_mean', 'condition_std', 'action_min', 'action_max'.
		"""
		return {
			"condition_mean": [self.condition_mean.tolist()],
			"condition_std": [self.condition_std.tolist()],
			"action_min": [self.action_min.tolist()],
			"action_max": [self.action_max.tolist()]
		}
	
	def save(self, filepath: str) -> None:
		"""
		Save normalization statistics to a Parquet file.

		Args:
			filepath (str): File path to save the statistics.
		"""
		data_dict = self.to_dict()
		table = pa.Table.from_pydict(data_dict)
		pq.write_table(table, filepath)
		print(f'Normalization stats saved to {filepath}')
		df = pd.read_parquet(filepath)
		for col in df.columns:
			df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, tuple, np.ndarray)) else x)
		print(tabulate(df, headers='keys', tablefmt='psql'))
	
	@classmethod
	def load(cls, filepath: str, device=None) -> "Normalize":
		"""
		Load normalization statistics from a Parquet file.
		If the file does not exist, return default stats.

		Args:
			filepath (str): Path to the Parquet file.
			device: Torch device.

		Returns:
			Normalize: Loaded normalization statistics instance.
		"""
		if not os.path.exists(filepath):
			print(f"Warning: '{filepath}' not found. Using default normalization stats.")
			default_condition_mean = torch.zeros(4, dtype=torch.float32, device=device)
			default_condition_std = torch.ones(4, dtype=torch.float32, device=device)
			default_action_min = torch.zeros(2, dtype=torch.float32, device=device)
			default_action_max = torch.ones(2, dtype=torch.float32, device=device)
			return cls(default_condition_mean, default_condition_std, default_action_min, default_action_max)
		table = pq.read_table(filepath)
		data = table.to_pydict()
		condition_mean = torch.tensor(data["condition_mean"][0], dtype=torch.float32, device=device)
		condition_std = torch.tensor(data["condition_std"][0], dtype=torch.float32, device=device)
		action_min = torch.tensor(data["action_min"][0], dtype=torch.float32, device=device)
		action_max = torch.tensor(data["action_max"][0], dtype=torch.float32, device=device)
		return cls(condition_mean, condition_std, action_min, action_max)

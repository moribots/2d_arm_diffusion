"""
Normalization module for computing, saving, and loading normalization statistics.
Uses Parquet for data storage.

For condition features, normalization is performed using mean and standard deviation.
For actions, min–max normalization is applied.

For actions, normalization is done as:
	   normalized_action = (action - action_min) / (action_max - action_min + eps)
and unnormalization as:
	   original_action = normalized_action * (action_max - action_min + eps) + action_min
"""

import torch
import pyarrow as pa
import pyarrow.parquet as pq
import os

class Normalize:
	def __init__(self, condition_mean, condition_std, action_min, action_max):
		"""
		Initialize normalization statistics.

		Args:
			condition_mean (Tensor): Mean vector for condition features.
			condition_std (Tensor): Standard deviation for condition features.
			action_min (Tensor): Minimum vector for actions.
			action_max (Tensor): Maximum vector for actions.
		"""
		self.condition_mean = condition_mean  # e.g. shape (4,)
		self.condition_std = condition_std
		self.action_min = action_min          # e.g. shape (2,)
		self.action_max = action_max

	@classmethod
	def compute_from_samples(cls, samples):
		"""
		Compute normalization statistics from samples.

		Args:
			samples (list): List of sample dictionaries.

		Returns:
			Normalize: Instance with computed statistics.
		"""
		conditions = []
		actions_list = []
		for sample in samples:
			if "observation" in sample and "state" in sample["observation"]:
				state = torch.tensor(sample["observation"]["state"], dtype=torch.float32)
				cond = state.flatten()  # Expected shape: (4,)
				conditions.append(cond)
			elif "observation.state" in sample:
				state = torch.tensor(sample["observation.state"], dtype=torch.float32)
				cond = state.flatten()
				conditions.append(cond)
			else:
				raise KeyError("Sample must contain observation['state']")
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

	def normalize_condition(self, condition):
		"""
		Normalize a condition tensor.

		If the number of elements in the stored condition stats does not match the last dimension
		of the input condition (e.g., stored stats have 2 elements but condition is 4-dimensional),
		then duplicate the stored stats.

		Args:
			condition (Tensor): Condition tensor (expected shape: (4,) or (B, 4)).

		Returns:
			Tensor: Normalized condition.
		"""
		# Determine expected dimension.
		if condition.dim() == 1:
			cond_dim = condition.shape[0]
		else:
			cond_dim = condition.shape[-1]

		if self.condition_mean.numel() != cond_dim:
			condition_mean = torch.cat([self.condition_mean, self.condition_mean], dim=0).unsqueeze(0)
			condition_std = torch.cat([self.condition_std, self.condition_std], dim=0).unsqueeze(0)
		else:
			if condition.dim() == 1:
				condition_mean = self.condition_mean
				condition_std = self.condition_std
			else:
				condition_mean = self.condition_mean.unsqueeze(0)
				condition_std = self.condition_std.unsqueeze(0)
		return (condition - condition_mean) / condition_std

	def normalize_action(self, action, eps=1e-6):
		"""
		Normalize an action tensor using min–max normalization.

		Args:
			action (Tensor): Raw action tensor.
			eps (float): Small epsilon to prevent division by zero.

		Returns:
			Tensor: Normalized action.
		"""
		return (action - self.action_min) / (self.action_max - self.action_min + eps)

	def unnormalize_action(self, action, eps=1e-6):
		"""
		Revert normalization on an action tensor.

		Args:
			action (Tensor): Normalized action tensor.
			eps (float): Small epsilon to prevent division by zero.

		Returns:
			Tensor: Original action tensor.
		"""
		return action * (self.action_max - self.action_min + eps) + self.action_min

	def to_dict(self):
		"""
		Convert statistics to a dictionary suitable for creating a single-row table.
		Each statistic is wrapped in a list so that all columns have equal length.
		"""
		return {
			"condition_mean": [self.condition_mean.tolist()],
			"condition_std": [self.condition_std.tolist()],
			"action_min": [self.action_min.tolist()],
			"action_max": [self.action_max.tolist()]
		}

	def save(self, filepath):
		"""
		Save normalization statistics to a Parquet file.

		Args:
			filepath (str): File path for saving.
		"""
		table = pa.Table.from_pydict(self.to_dict())
		pq.write_table(table, filepath)

	@classmethod
	def load(cls, filepath, device=None):
		"""
		Load normalization statistics from a Parquet file.
		If the file is not found, return default normalization stats.

		Args:
			filepath (str): Path to the Parquet file.
			device: Torch device.

		Returns:
			Normalize: Loaded normalization instance or a default instance.
		"""
		if not os.path.exists(filepath):
			print(f"Warning: Normalization stats file '{filepath}' not found. Using default normalization stats.")
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

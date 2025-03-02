"""
Normalization module for computing, saving, and loading normalization statistics.
Uses Parquet for data storage.
"""

import torch
import pyarrow as pa
import pyarrow.parquet as pq
import os

class Normalize:
	def __init__(self, condition_mean, condition_std, action_mean, action_std):
		"""
		Initialize normalization statistics.

		Args:
			condition_mean (Tensor): Mean vector for condition features.
			condition_std (Tensor): Standard deviation for condition features.
			action_mean (Tensor): Mean vector for actions.
			action_std (Tensor): Standard deviation for actions.
		"""
		self.condition_mean = condition_mean
		self.condition_std = condition_std
		self.action_mean = action_mean
		self.action_std = action_std

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
				cond = state.flatten()
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
		condition_std = conditions_cat.std(dim=0, unbiased=False) + 1e-6  # avoid division by zero.
		actions_cat = torch.cat(actions_list, dim=0)
		action_mean = actions_cat.mean(dim=0)
		action_std = actions_cat.std(dim=0) + 1e-6
		return cls(condition_mean, condition_std, action_mean, action_std)

	def normalize_condition(self, condition):
		"""
		Normalize a condition tensor.

		This method expects that 'condition' is a flat tensor.
		If self.condition_mean has shape (2,) but the input condition has shape (4,),
		we duplicate the stored stats.

		Args:
			condition (Tensor): Condition tensor (expected shape: (4,) or (B,4)).

		Returns:
			Tensor: Normalized condition.
		"""
		# If condition_mean has 2 elements (i.e. from previous runs where only a single state was stored)
		# and condition has 4 elements, duplicate the stats.
		if self.condition_mean.shape[0] == 2 and condition.shape[-1] == 4:
			condition_mean = torch.cat([self.condition_mean, self.condition_mean], dim=0).unsqueeze(0)
			condition_std = torch.cat([self.condition_std, self.condition_std], dim=0).unsqueeze(0)
		else:
			# Assume shapes match or condition is batched.
			if condition.dim() == 1:
				condition_mean = self.condition_mean
				condition_std = self.condition_std
			else:
				condition_mean = self.condition_mean.unsqueeze(0)
				condition_std = self.condition_std.unsqueeze(0)
		return (condition - condition_mean) / condition_std

	def normalize_action(self, action):
		"""
		Normalize an action tensor.
		"""
		return (action - self.action_mean) / self.action_std

	def unnormalize_action(self, action):
		"""
		Revert normalization on an action tensor.
		"""
		return action * self.action_std + self.action_mean

	def to_dict(self):
		"""
		Convert statistics to a dictionary.
		"""
		return {
			"condition_mean": self.condition_mean.tolist(),
			"condition_std": self.condition_std.tolist(),
			"action_mean": self.action_mean.tolist(),
			"action_std": self.action_std.tolist()
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
		If the file is not found, return default normalization stats (zeros for means and ones for stds).

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
			default_action_mean = torch.zeros(2, dtype=torch.float32, device=device)
			default_action_std = torch.ones(2, dtype=torch.float32, device=device)
			return cls(default_condition_mean, default_condition_std, default_action_mean, default_action_std)
		table = pq.read_table(filepath)
		data = table.to_pydict()
		condition_mean = torch.tensor(data["condition_mean"], dtype=torch.float32, device=device)
		condition_std = torch.tensor(data["condition_std"], dtype=torch.float32, device=device)
		action_mean = torch.tensor(data["action_mean"], dtype=torch.float32, device=device)
		action_std = torch.tensor(data["action_std"], dtype=torch.float32, device=device)
		return cls(condition_mean, condition_std, action_mean, action_std)

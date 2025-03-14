import torch
import pyarrow as pa
import pyarrow.parquet as pq
import os
import pandas as pd
from tabulate import tabulate
import numpy as np
from einops import rearrange
from src.config import *

class Normalize:
	"""
	Normalization module for computing, saving, and loading normalization statistics.
	
	For both condition features and actions, min-max normalization is applied.
	
	The computed statistics are registered as buffers to ensure consistency between training 
	and inference.
	"""
	def __init__(self, condition_min: torch.Tensor, condition_max: torch.Tensor, 
				 action_min: torch.Tensor, action_max: torch.Tensor):
		"""
		Initialize normalization statistics and register them as buffers.

		Args:
			condition_min (Tensor): Minimum vector for condition features.
			condition_max (Tensor): Maximum vector for condition features.
			action_min (Tensor): Minimum vector for actions.
			action_max (Tensor): Maximum vector for actions.
		"""
		self.register_buffer('condition_min', condition_min)
		self.register_buffer('condition_max', condition_max)
		self.register_buffer('action_min', action_min)
		self.register_buffer('action_max', action_max)
	
	def register_buffer(self, name: str, tensor: torch.Tensor):
		"""
		Register a tensor as a buffer. In a full nn.Module, this would use register_buffer.
		Here, we simulate that by setting an attribute.
		"""
		setattr(self, name, tensor)
	
	@classmethod
	def compute_from_limits(cls, device=None) -> "Normalize":
		"""
		Compute normalization statistics based on known environment limits.

		Args:
			device: Torch device.

		Returns:
			Normalize: Instance with computed statistics.
		"""
		# For X,Y coordinates in condition (matching action limits)
		# TODO(mrahme): once goal conditioning is added, I'll
		# need to add a total of 6 dims for t-1 and t [x, y, theta].
		condition_min = torch.tensor([0.0, 0.0], 
									 dtype=torch.float32, device=device)
		condition_max = torch.tensor([ACTION_LIM, ACTION_LIM], 
									 dtype=torch.float32, device=device)

		# The action normalization
		action_min = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
		action_max = torch.tensor([ACTION_LIM, ACTION_LIM], dtype=torch.float32, device=device)
		# cls is class method.
		return cls(condition_min, condition_max, action_min, action_max)
	
	def normalize_condition(self, condition: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
		"""
		Normalize a condition tensor using min-max normalization.
		Works with both 1D tensors [a,] and 2D tensors [a,b].
		Also supports inputs where the last dimension is a multiple of the condition dimension
		(e.g., concatenated state vectors).

		Args:
			condition (Tensor): Condition tensor, either 1D or 2D.
			eps (float): Epsilon to avoid division by zero.

		Returns:
			Tensor: Normalized condition with same dimensionality as input.
		"""
		# Save original shape for restoration later
		original_dim = condition.dim()
		original_size = condition.shape[-1] if original_dim > 0 else 0
		
		# Ensure condition is 2D for uniform processing
		if original_dim == 1:
			condition = rearrange(condition, 'd -> 1 d')  # Add batch dimension [d] -> [1, d]
		
		 # Handle case where the last dimension is a multiple of condition_min's size
		condition_dim = self.condition_min.shape[0]
		
		if condition.shape[-1] % condition_dim == 0:
			# Number of repeats needed
			repeat_factor = condition.shape[-1] // condition_dim
			
			# Repeat the min and max tensors to match the input dimension
			condition_min_expanded = self.condition_min.repeat(repeat_factor)
			condition_max_expanded = self.condition_max.repeat(repeat_factor)
			
			# Normalize from [condition_min, condition_max] to [-1, 1]
			normalized = 2.0 * ((condition - condition_min_expanded) / (condition_max_expanded - condition_min_expanded + eps)) - 1.0
		else:
			# Original normalization logic for exact dimension match
			normalized = 2.0 * ((condition - self.condition_min) / (self.condition_max - self.condition_min + eps)) - 1.0
		
		# Restore original shape if needed
		if original_dim == 1:
			normalized = rearrange(normalized, '1 d -> d')  # Remove batch dimension [1, d] -> [d]
			
		return normalized

	def unnormalize_condition(self, condition: torch.Tensor) -> torch.Tensor:
		"""
		Revert normalization on a condition tensor.
		Works with both 1D tensors [a,] and 2D tensors [a,b].
		Also supports inputs where the last dimension is a multiple of the condition dimension.

		Args:
			condition (Tensor): Normalized condition tensor, either 1D or 2D.

		Returns:
			Tensor: Original condition tensor with same dimensionality as input.
		"""
		# Save original shape for restoration later
		original_dim = condition.dim()
		original_size = condition.shape[-1] if original_dim > 0 else 0
		
		# Ensure condition is 2D for uniform processing
		if original_dim == 1:
			condition = rearrange(condition, 'd -> 1 d')  # Add batch dimension [d] -> [1, d]
		
		# Handle case where the last dimension is a multiple of condition_min's size
		condition_dim = self.condition_min.shape[0]
		
		if condition.shape[-1] % condition_dim == 0:
			# Number of repeats needed
			repeat_factor = condition.shape[-1] // condition_dim
			
			# Repeat the min and max tensors to match the input dimension
			condition_min_expanded = self.condition_min.repeat(repeat_factor)
			condition_max_expanded = self.condition_max.repeat(repeat_factor)
			
			# Un-normalize from [-1, 1] to [condition_min, condition_max]
			unnormalized = ((condition + 1.0) / 2.0) * (condition_max_expanded - condition_min_expanded) + condition_min_expanded
		else:
			# Original unnormalization logic for exact dimension match
			unnormalized = ((condition + 1.0) / 2.0) * (self.condition_max - self.condition_min) + self.condition_min
		
		# Restore original shape if needed
		if original_dim == 1:
			unnormalized = rearrange(unnormalized, '1 d -> d')  # Remove batch dimension [1, d] -> [d]
			
		return unnormalized

	def normalize_action(self, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
		"""
		Normalize an action tensor using min-max normalization.
		Works with both 1D tensors [a,] and 2D tensors [a,b].

		Args:
			action (Tensor): Raw action tensor, either 1D or 2D.
			eps (float): Epsilon to avoid division by zero.

		Returns:
			Tensor: Normalized action with same dimensionality as input.
		"""
		# Save original shape for restoration later
		original_dim = action.dim()
		
		# Ensure action is 2D for uniform processing
		if original_dim == 1:
			action = rearrange(action, 'd -> 1 d')  # Add batch dimension [d] -> [1, d]
		
		# Normalize from [action_min, action_max] to [-1, 1]
		normalized = 2.0 * ((action - self.action_min) / (self.action_max - self.action_min + eps)) - 1.0
		
		# Restore original shape if needed
		if original_dim == 1:
			normalized = rearrange(normalized, '1 d -> d')  # Remove batch dimension [1, d] -> [d]
			
		return normalized

	def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
		"""
		Revert normalization on an action tensor.
		Works with both 1D tensors [a,] and 2D tensors [a,b].

		Args:
			action (Tensor): Normalized action tensor, either 1D or 2D.

		Returns:
			Tensor: Original action tensor with same dimensionality as input.
		"""
		# Save original shape for restoration later
		original_dim = action.dim()
		
		# Ensure action is 2D for uniform processing
		if original_dim == 1:
			action = rearrange(action, 'd -> 1 d')  # Add batch dimension [d] -> [1, d]
		
		# Un-normalize from [-1, 1] to [action_min, action_max]
		unnormalized = ((action + 1.0) / 2.0) * (self.action_max - self.action_min) + self.action_min
		
		# Restore original shape if needed
		if original_dim == 1:
			unnormalized = rearrange(unnormalized, '1 d -> d')  # Remove batch dimension [1, d] -> [d]
			
		return unnormalized
	
	def to_dict(self) -> dict:
		"""
		Convert normalization statistics to a dictionary for saving.

		Returns:
			dict: Dictionary with keys 'condition_min', 'condition_max', 'action_min', 'action_max'.
		"""
		return {
			"condition_min": [self.condition_min.tolist()],
			"condition_max": [self.condition_max.tolist()],
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
			return cls.compute_from_limits(device)
			
		try:
			table = pq.read_table(filepath)
			data = table.to_pydict()
			
			condition_min = torch.tensor(data["condition_min"][0], dtype=torch.float32, device=device)
			condition_max = torch.tensor(data["condition_max"][0], dtype=torch.float32, device=device)
			action_min = torch.tensor(data["action_min"][0], dtype=torch.float32, device=device)
			action_max = torch.tensor(data["action_max"][0], dtype=torch.float32, device=device)
			# cls is class method.
			return cls(condition_min, condition_max, action_min, action_max)
		except Exception as e:
			print(f"Error loading normalization file: {e}")
			print("Using default normalization stats.")
			return cls.compute_from_limits(device)
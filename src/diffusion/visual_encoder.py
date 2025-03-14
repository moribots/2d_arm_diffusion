"""
VisualEncoder extracts keypoint features from an image.
Uses a pretrained ResNet18 backbone with GroupNorm and applies spatial softmax to obtain coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import resnet18, ResNet18_Weights
from typing import Callable
from einops import rearrange


def replace_submodules(
		root_module: nn.Module,
		predicate: Callable[[nn.Module], bool],
		func: Callable[[nn.Module], nn.Module]) -> nn.Module:
	"""
	Replace all submodules selected by the predicate with the output of func.
	
	Args:
		root_module (nn.Module): The root module to modify
		predicate (Callable): Returns true if the module is to be replaced
		func (Callable): Returns new module to use
		
	Returns:
		nn.Module: Modified module with replacements
	"""
	# Handle the case where the root module itself matches the predicate
	if predicate(root_module):
		return func(root_module)
	
	# Find all modules matching the predicate with their full paths
	# Collect all targets in one pass through the module tree
	modules_to_replace = {name: m for name, m in root_module.named_modules() 
						  if predicate(m) and name}  # Skip unnamed root
	
	# Replace each matching module
	for full_name, module in modules_to_replace.items():
		# get parent path and child name
		parent_name, _, name = full_name.rpartition('.')
		
		# Navigate to the parent module
		parent_module = root_module
		if parent_name:
			parent_module = root_module.get_submodule(parent_name)
			
		# Replace the module in the parent
		if isinstance(parent_module, nn.Sequential):
			parent_module[int(name)] = func(module)
		else:
			setattr(parent_module, name, func(module))
	
	# Verify all modules were replaced
	remaining = any(predicate(m) for name, m in root_module.named_modules() if name)
	assert not remaining, "Some modules were not replaced"
	
	return root_module


def replace_bn_with_gn(
	root_module: nn.Module,
	features_per_group: int=16) -> nn.Module:
	"""
	Replace all BatchNorm layers with GroupNorm.
	
	Args:
		root_module (nn.Module): Module to modify
		features_per_group (int): Number of features per GroupNorm group
		
	Returns:
		nn.Module: Modified module with GroupNorm layers
	"""
	replace_submodules(
		root_module=root_module,
		predicate=lambda x: isinstance(x, nn.BatchNorm2d),
		func=lambda x: nn.GroupNorm(
			num_groups=max(1, x.num_features//features_per_group),
			num_channels=x.num_features)
	)
	return root_module


class VisualEncoder(nn.Module):
	"""
	VisualEncoder extracts image features.
	
	- Loads a pretrained ResNet18 (excluding the classification head).
	- Replaces all BatchNorm layers with GroupNorm for better performance with small batches.
	- Applies a 1x1 convolution to reduce channels from 512 to 32.
	- Applies spatial softmax to compute expected x and y coordinates for each channel,
	  resulting in a feature vector of shape (B, 64).
	"""
	def __init__(self):
		super(VisualEncoder, self).__init__()
		# Load pretrained ResNet18
		resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		# Remove the final fully connected layer and avgpool layer
		# Because we are doing feature extraction, not classification
		resnet.fc = torch.nn.Identity()
		# replace BatchNorm with GroupNorm. Benefits:
		# - batch size variance robustness
		# - more stable gradients during training
		# - better generalization for small datasets
		# - behaves identically during inference and training
		resnet = replace_bn_with_gn(resnet)
		
		# Extract feature layers (exclude avgpool and fc)
		self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
		
		# 1x1 convolution to reduce channel dimension
		self.conv_reduce = nn.Conv2d(512, 32, kernel_size=1)

	def spatial_softmax(self, feature_map):
		"""
		Apply spatial softmax to extract keypoint coordinates.
		
		Args:
			feature_map (Tensor): Feature map of shape (B, C, H, W).
			
		Returns:
			Tensor: Keypoint coordinates of shape (B, C*2).
		"""
		B, C, H, W = feature_map.size()

		# Create a normalized coordinate grid spanning from -1 to 1 in both dimensions
		# pos_x will be a 2D tensor of shape (W, H) where each row has the same x-coordinate value
		# pos_y will be a 2D tensor of shape (W, H) where each column has the same y-coordinate value
		# indexing='ij' ensures pos_x varies along the first dimension and pos_y along the second
		# This grid is used as the spatial reference for computing weighted positions via spatial softmax
		pos_x, pos_y = torch.meshgrid(
			torch.linspace(-1.0, 1.0, W, device=feature_map.device),  # x-coordinates from -1 to 1 across width
			torch.linspace(-1.0, 1.0, H, device=feature_map.device),  # y-coordinates from -1 to 1 across height
			indexing='ij'  # Use 'ij' indexing for consistency with subsequent operations
		)
		
		# Reshape coordinate grids to 1D vectors using einops
		pos_x = rearrange(pos_x.t(), 'h w -> (h w)')  # [H*W]
		pos_y = rearrange(pos_y.t(), 'h w -> (h w)')  # [H*W]
		
		# Flatten the feature map with rearrange
		feature_flat = rearrange(feature_map, 'b c h w -> b c (h w)')
		softmax_attention = F.softmax(feature_flat, dim=-1)
		
		# 'b c n, n -> b c' computes weighted sum over spatial dimensions
		exp_x = torch.einsum('b c n, n -> b c', softmax_attention, pos_x)
		exp_y = torch.einsum('b c n, n -> b c', softmax_attention, pos_y)
		
		# Concatenate expected x and y
		keypoints = torch.cat([exp_x, exp_y], dim=1)
		return keypoints

	def forward(self, x):
		"""
		Forward pass of the visual encoder.
		
		Args:
			x (Tensor): Input image tensor of shape (B, 3, IMG_RES, IMG_RES).
			
		Returns:
			Tensor: Feature vector of shape (B, 64).
		"""
		features = self.feature_extractor(x)  # Extract convolutional features
		features = self.conv_reduce(features)   # Reduce channel dimension
		keypoints = self.spatial_softmax(features)  # Compute keypoint coordinates
		return keypoints

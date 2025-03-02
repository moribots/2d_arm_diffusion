"""
VisualEncoder extracts keypoint features from an image.
Uses a pretrained ResNet18 backbone and applies spatial softmax to obtain coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

class VisualEncoder(nn.Module):
	"""
	VisualEncoder extracts image features.
	
	- Loads a pretrained ResNet18 (excluding the classification head).
	- Applies a 1x1 convolution to reduce channels from 512 to 32.
	- Applies spatial softmax to compute expected x and y coordinates for each channel,
	  resulting in a feature vector of shape (B, 64).
	"""
	def __init__(self):
		super(VisualEncoder, self).__init__()
		# Load pretrained ResNet18.
		resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
		# 1x1 convolution to reduce channel dimension.
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
		# Create a coordinate grid.
		pos_x, pos_y = torch.meshgrid(
			torch.linspace(-1.0, 1.0, W, device=feature_map.device),
			torch.linspace(-1.0, 1.0, H, device=feature_map.device),
			indexing='ij'
		)
		pos_x = pos_x.t().reshape(1, 1, H, W)
		pos_y = pos_y.t().reshape(1, 1, H, W)
		# Flatten the feature map.
		feature_map = feature_map.view(B, C, -1)
		softmax_attention = F.softmax(feature_map, dim=-1)
		# Rearrange coordinate grids.
		pos_x = rearrange(pos_x, '1 1 h w -> 1 1 (h w)')
		pos_y = rearrange(pos_y, '1 1 h w -> 1 1 (h w)')
		exp_x = torch.sum(softmax_attention * pos_x, dim=-1)
		exp_y = torch.sum(softmax_attention * pos_y, dim=-1)
		# Concatenate expected x and y.
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
		features = self.feature_extractor(x)  # Extract convolutional features.
		features = self.conv_reduce(features)   # Reduce channel dimension.
		keypoints = self.spatial_softmax(features)  # Compute keypoint coordinates.
		return keypoints

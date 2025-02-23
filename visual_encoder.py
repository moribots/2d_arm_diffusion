import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisualEncoder(nn.Module):
	"""
	Visual encoder that extracts image features using a modified ResNet18 and spatial softmax.
	
	- Uses a pretrained ResNet18 backbone.
	- Applies a 1x1 convolution to reduce the channel dimension to 16.
	- Applies spatial softmax to obtain 2D keypoint coordinates per channel, resulting in 32-dimensional output.

	Inspired by the LeRobot implementation and the paper "Spatial Softmax for Visual Attention" (https://arxiv.org/abs/1706.03762).
	- The output is a tensor of shape (B, 32), where B is the batch size.
	- The spatial softmax is applied to the feature map to obtain expected x and y coordinates for each channel.
	- The output is used as part of the conditioning input for the diffusion policy model.
	- The model is designed to work with images of size (3, IMG_RES, IMG_RES).
	"""
	def __init__(self):
		super(VisualEncoder, self).__init__()
		# Load pretrained ResNet18 backbone
		resnet = models.resnet18(pretrained=True)
		# Use all layers except the fully connected layer
		self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Output shape: (B, 512, H, W)
		
		# 1x1 convolution to reduce channels from 512 to 16
		self.conv_reduce = nn.Conv2d(512, 16, kernel_size=1)
		
	def spatial_softmax(self, feature_map):
		"""
		Apply spatial softmax to the feature map.
		
		feature_map: Tensor of shape (B, C, H, W)
		Returns: Tensor of shape (B, C*2) containing expected x and y coordinates for each channel.
		"""
		B, C, H, W = feature_map.size()
		# Create coordinate grid
		pos_x, pos_y = torch.meshgrid(torch.linspace(-1.0, 1.0, W, device=feature_map.device),
									  torch.linspace(-1.0, 1.0, H, device=feature_map.device))
		pos_x = pos_x.t().reshape(1, 1, H, W)
		pos_y = pos_y.t().reshape(1, 1, H, W)
		
		# Flatten spatial dimensions
		feature_map = feature_map.view(B, C, -1)
		softmax_attention = F.softmax(feature_map, dim=-1)
		
		# Reshape coordinate grids to (1, 1, H*W)
		pos_x = pos_x.view(1, 1, -1)
		pos_y = pos_y.view(1, 1, -1)
		
		exp_x = torch.sum(softmax_attention * pos_x, dim=-1)
		exp_y = torch.sum(softmax_attention * pos_y, dim=-1)
		# Concatenate expected coordinates
		keypoints = torch.cat([exp_x, exp_y], dim=1)  # (B, 2C)
		return keypoints

	def forward(self, x):
		"""
		Forward pass to extract image features.
		
		x: Input image tensor of shape (B, 3, IMG_RES, IMG_RES)
		Returns: Feature vector of shape (B, 32)
		"""
		features = self.feature_extractor(x)  # (B, 512, H, W)
		features = self.conv_reduce(features)   # (B, 16, H, W)
		keypoints = self.spatial_softmax(features)  # (B, 32)
		return keypoints

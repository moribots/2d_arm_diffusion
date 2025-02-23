import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

class VisualEncoder(nn.Module):
	"""
	Visual encoder that extracts image features using a modified ResNet18 and spatial softmax.
	
	- Uses a pretrained ResNet18 backbone.
	- Applies a 1x1 convolution to reduce the channel dimension to 16.
	- Applies spatial softmax to obtain 2D keypoint coordinates per channel, resulting in 32-dimensional output.
	"""
	def __init__(self):
		super(VisualEncoder, self).__init__()
		# Load pretrained ResNet18 backbone
		resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
		pos_x, pos_y = torch.meshgrid(
			torch.linspace(-1.0, 1.0, W, device=feature_map.device),
			torch.linspace(-1.0, 1.0, H, device=feature_map.device),
			indexing='ij'
		)
		pos_x = pos_x.t().reshape(1, 1, H, W)
		pos_y = pos_y.t().reshape(1, 1, H, W)
		
		# Flatten spatial dimensions of feature_map
		feature_map = feature_map.view(B, C, -1)
		softmax_attention = F.softmax(feature_map, dim=-1)
		
		# Use einops to reshape coordinate grids to (1, 1, H*W)
		pos_x = rearrange(pos_x, '1 1 h w -> 1 1 (h w)')
		pos_y = rearrange(pos_y, '1 1 h w -> 1 1 (h w)')
		
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

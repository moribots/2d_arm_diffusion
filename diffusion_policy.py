import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
	"""
	A residual block for MLPs that applies layer normalization,
	a hidden linear transformation with GELU activation, dropout, and a residual connection.
	"""
	def __init__(self, in_features, hidden_features, dropout=0.0):
		super().__init__()
		self.norm = nn.LayerNorm(in_features)
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.activation = nn.GELU()
		self.fc2 = nn.Linear(hidden_features, in_features)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		residual = x
		x = self.norm(x)
		x = self.fc1(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.dropout(x)
		return x + residual

class DiffusionPolicy(nn.Module):
	def __init__(self, action_dim, condition_dim, time_embed_dim=128):
		"""
		Diffusion policy network.

		This network is used in the reverse diffusion process to predict the noise that
		was added to an action sample. In our project, the action is the end-effector (EE)
		position (a 2D vector), and the condition is the desired object pose [x, y, theta].

		- GELU activations instead of ReLU for smoother gradients.
		- A time embedding MLP that converts the scalar timestep into a higher-dimensional vector.
		- Residual blocks with layer normalization to improve gradient flow.
		 

		Parameters:
		- action_dim (int): 2
		- condition_dim (int): 3
		- time_embed_dim (int): Dimension of the embedding space for the diffusion timestep.

		The network takes as input:
		 - a noised action sample,
		 - the timestep at which the sample is taken,
		 - a conditioning signal (desired T pose).

		Output: an estimate of the noise that was added to the action sample.
		"""
		super(DiffusionPolicy, self).__init__()

		# Time Embedding:
		# The diffusion process adds timestep-dependent noise. The model needs to know the current timestep to
		# properly denoise the sample. We use a small MLP to convert the scalar timestep
		# into a higher-dimensional vector (the time embedding). This embedding allows the network
		# to modulate its behavior based on how much noise has been added.
		self.time_embed = nn.Sequential(
			nn.Linear(1, time_embed_dim),
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim),
			nn.GELU()
		)

		# Main Network:
		# Takes as input the concatenation of:
		# - the noised action sample (x),
		# - the conditioning signal (desired T pose),
		# - the time embedding.
		# This combined vector is processed to predict the noise added to the action sample.

		# Instead of a simple MLP, we use an initial linear layer followed by residual MLP blocks.
		# Input dimension = action_dim + condition_dim + time_embed_dim.
		in_features = action_dim + condition_dim + time_embed_dim
		self.fc_initial = nn.Linear(in_features, 256)
		self.res_block1 = ResidualMLPBlock(256, 256, dropout=0.1)
		self.res_block2 = ResidualMLPBlock(256, 256, dropout=0.1)
		# Final output layer maps to action_dim.
		self.fc_out = nn.Linear(256, action_dim)

	def forward(self, x, t, condition):
		"""
		Forward pass through the diffusion policy network.

		Parameters:
		- x: Noised action tensor of shape (batch, action_dim).
			 This represents the EE position sample with added noise.
		- t: Timestep tensor of shape (batch,). It indicates the diffusion timestep
			 corresponding to the noise level in x.
		- condition: Conditioning signal tensor of shape (batch, condition_dim).
			 This represents the desired T pose.

		Process:
		1. t is reshaped to (batch, 1) and converted to float.
		2. The time embedding network transforms t into a higher-dimensional vector.
		3. The noised action, conditioning signal, and time embedding are concatenated.
		4. The MLP processes this input to output an estimate of the added noise.

		Explanation of input and output shapes:
		- x: (batch, action_dim)
		- t: (batch,)
		- condition: (batch, condition_dim)
		- t reshaped: (batch, 1)
		- t_emb: (batch, time_embed_dim)
		- x_in: (batch, action_dim + condition_dim + time_embed_dim)
		- noise_pred: (batch, action_dim)

		Returns:
		- noise_pred: The predicted noise vector of shape (batch, action_dim).
		"""

		# Reshape t to (batch, 1)
		t = rearrange(t, 'b -> b 1').float()
		# Obtain the time embedding for the given timestep.
		t_emb = self.time_embed(t)  # (batch, time_embed_dim)
		# Concatenate the noised action x, the condition (desired T pose), and the time embedding.
		# Could not use einops here because each of the tensors have different dimensions.
		x_in = torch.cat([x, condition, t_emb], dim=-1)  # (batch, action_dim + condition_dim + time_embed_dim)
		
		x = self.fc_initial(x_in)
		x = F.gelu(x)
		x = self.res_block1(x)
		x = self.res_block2(x)
		noise_pred = self.fc_out(x)
		return noise_pred
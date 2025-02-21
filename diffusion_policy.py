import torch
from einops import rearrange
import torch.nn as nn

class DiffusionPolicy(nn.Module):
	def __init__(self, action_dim, condition_dim, time_embed_dim=128):
		"""
		Diffusion policy network.

		This network is used in the reverse diffusion process to predict the noise that
		was added to an action sample. In our project, the action is the end-effector (EE)
		position (a 2D vector), and the condition is the desired object pose [x, y, theta].

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
			nn.Linear(1, time_embed_dim),  # Embed the scalar timestep into a vector.
			nn.ReLU(),
			nn.Linear(time_embed_dim, time_embed_dim),  # Process further to capture complex patterns.
		)

		# Main Network:
		# The main network is an MLP that takes as input the concatenation of:
		# - the noised action sample (x),
		# - the conditioning signal (desired T pose),
		# - the time embedding.
		# This combined vector is processed to predict the noise added to the action sample.
		self.model = nn.Sequential(
			nn.Linear(action_dim + condition_dim + time_embed_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim)  # Output matches the action dimension.
		)

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
		# Reshape timestep tensor to (batch, 1).
		t = rearrange(t, 'b -> b 1').float()
		# Obtain the time embedding for the given timestep.
		t_emb = self.time_embed(t)  # Shape: (batch, time_embed_dim)
		# Concatenate the noised action x, the condition (desired T pose), and the time embedding.
		# Could not use einops here because each of the tensors have different dimensions.
		x_in = torch.cat([x, condition, t_emb], dim=-1)

		# Forward pass through the MLP.
		noise_pred = self.model(x_in)  # Shape: (batch, action_dim)
		return noise_pred
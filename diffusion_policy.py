import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from config import WINDOW_SIZE

class ResidualMLPBlock(nn.Module):
	"""
	A residual block for MLPs that applies layer normalization,
	a hidden linear transformation with GELU activation, dropout, and a residual connection.
	This is used to help with gradient flow in deeper architectures.
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
	def __init__(self, action_dim, condition_dim, time_embed_dim=128, window_size=WINDOW_SIZE+1):
		"""
		Diffusion policy network.

		This network is used in the reverse diffusion process to predict the noise that
		was added to an action sample. In our project, the action is the end-effector (EE)
		position (a 2D vector) and we now consider a temporal window of actions. The window now
		includes the timestep t-1 as well as t to t+(window_size-1), so the input window length
		is (WINDOW_SIZE+1).

		- GELU activations instead of ReLU for smoother gradients.
		- A time embedding MLP that converts the scalar timestep into a higher-dimensional vector.
		- Residual blocks with layer normalization to improve gradient flow.

		Parameters:
		- action_dim (int): Dimension of the action (e.g., 2)
		- condition_dim (int): Dimension of the conditioning signal (e.g., 9)
		- time_embed_dim (int): Dimension of the embedding space for the diffusion timestep.
		- window_size (int): Number of consecutive action timestamps to consider (includes t-1).
		
		The network takes as input:
		- a noised action sample of shape (batch, window_size, action_dim),
		- the timestep at which the sample is taken,
		- a conditioning signal (desired object pose and previous/current T poses).

		Output: an estimate of the noise that was added to the entire window, with shape (batch, window_size, action_dim).
		"""
		super(DiffusionPolicy, self).__init__()
		self.window_size = window_size

		# Time Embedding:
		# The diffusion process adds timestep-dependent noise. The model needs to know the current timestep to
		# properly denoise the sample. We use a small MLP to convert the scalar timestep into a higher-dimensional
		# vector (the time embedding). This embedding allows the network to modulate its behavior based on how much noise has been added.
		self.time_embed = nn.Sequential(
			nn.Linear(1, time_embed_dim), # Embed the scalar timestep into a vector.
			nn.GELU(),
			nn.Linear(time_embed_dim, time_embed_dim), # Process further to capture complex patterns.
			nn.GELU()
		)

		# Main Network:
		# The main network is an MLP that takes as input the concatenation of:
		# - the flattened noised action sample (x) with shape (batch, window_size * action_dim),
		# - the conditioning signal (desired T pose) with shape (batch, condition_dim),
		# - the time embedding with shape (batch, time_embed_dim).
		# This combined vector is processed by an initial linear layer, followed by two residual MLP blocks,
		# and finally mapped to the output predicting the noise for the entire window.
		in_features = window_size * action_dim + condition_dim + time_embed_dim
		out_features = window_size * action_dim # Predict noise for every timestep in the window.
		self.fc_initial = nn.Linear(in_features, 256)
		self.res_block1 = ResidualMLPBlock(256, 256, dropout=0.1)
		self.res_block2 = ResidualMLPBlock(256, 256, dropout=0.1)
		self.fc_out = nn.Linear(256, out_features)

	def forward(self, x, t, condition):
		"""
		Forward pass through the diffusion policy network.

		Parameters:
		- x: Noised action tensor of shape (batch, window_size, action_dim).
			  This represents a temporal window of EE position samples (including the timestep t-1) with added noise.
		- t: Timestep tensor of shape (batch,). It indicates the diffusion timestep corresponding to the noise level in x.
		- condition: Conditioning signal tensor of shape (batch, condition_dim).
					 This represents the desired object pose along with the previous and current T poses.

		Process:
		1. Reshape t to (batch, 1) and convert to float.
		2. Compute the time embedding from t.
		3. Flatten the temporal window of actions from (batch, window_size, action_dim) to (batch, window_size * action_dim).
		4. Concatenate the flattened noised action, the conditioning signal, and the time embedding.
		5. Process this input through an initial linear layer, two residual blocks, and a final output layer.
		6. Reshape the output back to (batch, window_size, action_dim).

		Returns:
		- noise_pred: The predicted noise tensor of shape (batch, window_size, action_dim).
		"""
		# If x is 2D, add a temporal dimension.
		if x.dim() == 2:
			x = x.unsqueeze(1)  # Now x shape becomes (batch, 1, action_dim)

		# Reshape timestep tensor to (batch, 1)
		t = rearrange(t, 'b -> b 1').float()
		t_emb = self.time_embed(t)  # (batch, time_embed_dim)

		# Flatten the temporal window of actions.
		x_flat = rearrange(x, 'b w a -> b (w a)')  # (batch, window_size * action_dim)

		# Concatenate the flattened noised action, the condition, and the time embedding.
		x_in = torch.cat([x_flat, condition, t_emb], dim=-1)

		# Forward pass through the network.
		x_out = self.fc_initial(x_in)
		x_out = F.gelu(x_out)
		x_out = self.res_block1(x_out)
		x_out = self.res_block2(x_out)
		noise_pred_flat = self.fc_out(x_out)
		# Reshape the output back to the temporal window shape.
		noise_pred = rearrange(noise_pred_flat, 'b (w a) -> b w a', w=self.window_size, a=x.size(-1))
		return noise_pred

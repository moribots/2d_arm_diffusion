import os
import json
import csv  # For CSV dumps of training metrics.
import math  # Needed for cosine annealing if desired.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization.
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from config import TRAINING_DATA_DIR, T, BATCH_SIZE, EPOCHS, LEARNING_RATE, ACTION_DIM, CONDITION_DIM, WINDOW_SIZE
from einops import rearrange

# ------------------------- Dataset Definition -------------------------
class PolicyDataset(Dataset):
	"""
	PolicyDataset loads training samples from JSON files stored in TRAINING_DATA_DIR.
	Each sample is expected to have:
	  - "goal_pose": The desired T pose [x, y, theta].
	  - "T_pose": The current T object pose [x, y, theta].
	  - "action": The end-effector (EE) position (a 2D vector) recorded during data collection.
				For temporal data, this may be a list of actions (each a 2D vector).
	The condition is the concatenation of goal_pose and T_pose, forming a 6D vector.
	"""
	def __init__(self, data_dir):
		self.samples = []
		for filename in os.listdir(data_dir):
			if filename.endswith('.json'):
				filepath = os.path.join(data_dir, filename)
				with open(filepath, 'r') as f:
					data = json.load(f)
					self.samples.extend(data)
	
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		sample = self.samples[idx]
		goal_pose = torch.tensor(sample["goal_pose"], dtype=torch.float32)  # (3,)
		T_pose = torch.tensor(sample["T_pose"], dtype=torch.float32)        # (3,)
		condition = torch.cat([goal_pose, T_pose], dim=0)  # (6,)
		# Modified logic: Support both single action and sequence of actions.
		action_data = sample["action"]
		if isinstance(action_data[0], (list, tuple)):
			action = torch.tensor(action_data, dtype=torch.float32)  # (seq_len, action_dim)
		else:
			action = torch.tensor(action_data, dtype=torch.float32)  # (action_dim,)
		return condition, action

# ------------------------- Training Loop -------------------------
def train():
	print(f"CUDA is available: {torch.cuda.is_available()}")
	# ------------------------- Visualization and CSV Setup -------------------------
	writer = SummaryWriter(log_dir="runs/diffusion_policy_training")
	csv_file = open("training_metrics.csv", "w", newline="")
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(["epoch", "avg_loss"])
	# -------------------------------------------------------------------------
	
	torch.backends.cudnn.benchmark = True
	
	dataset = PolicyDataset(TRAINING_DATA_DIR)
	dataloader = DataLoader(
		dataset, 
		batch_size=BATCH_SIZE, 
		shuffle=True, 
		num_workers=4,
		pin_memory=True
	)
	
	# Initialize the diffusion policy model.
	# Ensure that your model is instantiated with the correct window_size.
	model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE)
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
	# We use elementwise loss to allow weighting.
	mse_loss = nn.MSELoss(reduction="none")
	
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)
	
	betas = get_beta_schedule(T)
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)
	
	for epoch in range(EPOCHS):
		running_loss = 0.0
		
		for condition, action in dataloader:
			condition = condition.to(device)  # (batch, 6)
			action = action.to(device)        # (batch, 2) or (batch, seq_len, 2)
			
			# Modified logic: Create a temporal window of actions.
			if action.ndim == 2:
				# Create a temporal window from the single action by repeating along a new dimension.
				action_seq = action.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)  # (batch, WINDOW_SIZE, action_dim)
			elif action.ndim == 3:
				seq_len = action.shape[1]
				if seq_len < WINDOW_SIZE:
					# Pad by repeating the last action.
					pad = action[:, -1:, :].repeat(1, WINDOW_SIZE - seq_len, 1)
					action_seq = torch.cat([action, pad], dim=1)
				else:
					# Truncate if longer than WINDOW_SIZE.
					action_seq = action[:, :WINDOW_SIZE, :]
			else:
				raise ValueError("Unexpected shape for action tensor")
			
			# Sample a diffusion timestep 't' for each sample.
			t = torch.randint(0, T, (action_seq.size(0),), device=device)
			# Reshape alpha_bar to be broadcastable to (batch, WINDOW_SIZE, action_dim)
			alpha_bar = rearrange(alphas_cumprod[t], 'b -> b 1 1')
			
			# Generate noise for the entire temporal window.
			noise = torch.randn_like(action_seq)
			
			# Apply the forward diffusion process:
			# x_t = sqrt(alpha_bar)*x_0 + sqrt(1 - alpha_bar)*noise.
			x_t = torch.sqrt(alpha_bar) * action_seq + torch.sqrt(1 - alpha_bar) * noise
			
			# Forward pass through the model.
			# x_t now has shape: (batch, WINDOW_SIZE, action_dim)
			noise_pred = model(x_t, t.float(), condition)
			
			# --- Loss Weighting ---
			# Weight the loss by sqrt(1 - alpha_bar) per sample (broadcasted to the window and action dims).
			weight = torch.sqrt(1 - alpha_bar)
			loss_elements = mse_loss(noise_pred, noise)  # Shape: (batch, WINDOW_SIZE, action_dim)
			loss = torch.mean(weight * loss_elements)
			# --- End Loss Weighting ---
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item() * action_seq.size(0)
		
		avg_loss = running_loss / len(dataset)
		print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
		
		# ------------------------- Visualization and CSV Dump -------------------------
		writer.add_scalar("Loss/avg_loss", avg_loss, epoch+1)
		# Log current learning rate.
		current_lr = optimizer.param_groups[0]['lr']
		writer.add_scalar("LearningRate", current_lr, epoch+1)
		# Log weight histograms.
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch+1)
		# Log gradient norms.
		for name, param in model.named_parameters():
			if param.grad is not None:
				grad_norm = param.grad.data.norm(2).item()
				writer.add_scalar(f"Gradients/{name}_norm", grad_norm, epoch+1)
		csv_writer.writerow([epoch+1, avg_loss])
		# -------------------------------------------------------------------------
		
		# Overwrite the trained policy every 100 epochs.
		if (epoch + 1) % 100 == 0:
			torch.save(model.state_dict(), "diffusion_policy.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")
	
	torch.save(model.state_dict(), "diffusion_policy.pth")
	print("Training complete. Model saved as diffusion_policy.pth.")
	
	writer.close()
	csv_file.close()

if __name__ == "__main__":
	train()

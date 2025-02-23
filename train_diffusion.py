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
from normalize import Normalize  # New normalization helper class

# ------------------------- Dataset Definition -------------------------
class PolicyDataset(Dataset):
	"""
	PolicyDataset loads training samples from JSON files stored in TRAINING_DATA_DIR.
	
	Each sample is expected to have:
	  - "observation": a dict with:
			- "image": list of image filenames (e.g. [img_{t-1}, img_t])
			- "state": list of two states (each a 2D vector, e.g. [[x1,y1], [x2,y2]])
	  - "action": a list of actions (each a 2D vector) representing the trajectory from t-1 to t+14.
	
	The condition is built by flattening the observation["state"] (resulting in 4 numbers).
	Normalization of conditions and actions is handled via the Normalize class.
	"""
	def __init__(self, data_dir):
		self.samples = []
		for filename in os.listdir(data_dir):
			if filename.endswith('.json'):
				filepath = os.path.join(data_dir, filename)
				with open(filepath, 'r') as f:
					data = json.load(f)
					self.samples.extend(data)
		# Compute normalization stats using the Normalize class.
		self.normalize = Normalize.compute_from_samples(self.samples)
		# Optionally, save the normalization stats to a file for later use.
		self.normalize.save("normalization_stats.json")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		# Build condition from observation["state"].
		if "observation" not in sample or "state" not in sample["observation"]:
			raise KeyError("Sample must contain observation['state']")
		state = torch.tensor(sample["observation"]["state"], dtype=torch.float32)  # shape (2,2)
		condition = state.flatten()  # (4,)
		# Normalize condition using the Normalize class.
		condition = self.normalize.normalize_condition(condition)

		if "action" not in sample:
			raise KeyError("Sample does not contain 'action'")
		action_data = sample["action"]
		if isinstance(action_data[0], (list, tuple)):
			action = torch.tensor(action_data, dtype=torch.float32)  # (seq_len, action_dim)
		else:
			action = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)  # (1, action_dim)
		# Normalize action using the Normalize class.
		action = self.normalize.normalize_action(action)
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
	# Initialize the diffusion policy model with the updated window size.
	model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE+1)
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
	# Use a cosine annealing scheduler to match LeRobot training.
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
	mse_loss = nn.MSELoss(reduction="none")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)
	betas = get_beta_schedule(T)
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)

	for epoch in range(EPOCHS):
		running_loss = 0.0
		for condition, action in dataloader:
			condition = condition.to(device)  # (batch, 4)
			action = action.to(device)  # (batch, action_dim) or (batch, seq_len, action_dim)

			# Create a temporal window of actions including t-1.
			# LeRobot uses a window from t-1 to t+WINDOW_SIZE-1.
			target_length = WINDOW_SIZE + 1
			if action.ndim == 2:
				# Duplicate the single action to form a window.
				action_seq = action.unsqueeze(1).repeat(1, target_length, 1)  # (batch, target_length, action_dim)
			elif action.ndim == 3:
				seq_len = action.shape[1]
				if seq_len < target_length:
					# Pad by repeating the last action.
					pad = action[:, -1:, :].repeat(1, target_length - seq_len, 1)
					action_seq = torch.cat([action, pad], dim=1)
				else:
					# Truncate if longer than target_length.
					action_seq = action[:, :target_length, :]
			else:
				raise ValueError("Unexpected shape for action tensor")

			# Sample a diffusion timestep 't' for each sample.
			t = torch.randint(0, T, (action_seq.size(0),), device=device)
			# Reshape alphas_cumprod[t] to be broadcastable to (batch, target_length, action_dim)
			alpha_bar = rearrange(alphas_cumprod[t], 'b -> b 1 1')
			# Generate noise for the entire temporal window.
			noise = torch.randn_like(action_seq)
			# Apply the forward diffusion process:
			# x_t = sqrt(alpha_bar)*x_0 + sqrt(1 - alpha_bar)*noise.
			x_t = torch.sqrt(alpha_bar) * action_seq + torch.sqrt(1 - alpha_bar) * noise

			# Forward pass through the model.
			noise_pred = model(x_t, t.float(), condition)
			# --- Loss Weighting ---
			weight = torch.sqrt(1 - alpha_bar)
			loss_elements = mse_loss(noise_pred, noise)  # Shape: (batch, target_length, action_dim)
			loss = torch.mean(weight * loss_elements)
			# --- End Loss Weighting --

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * action_seq.size(0)

		avg_loss = running_loss / len(dataset)
		print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
		writer.add_scalar("Loss/avg_loss", avg_loss, epoch+1)
		current_lr = optimizer.param_groups[0]['lr']
		writer.add_scalar("LearningRate", current_lr, epoch+1)
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch+1)
		for name, param in model.named_parameters():
			if param.grad is not None:
				grad_norm = param.grad.data.norm(2).item()
				writer.add_scalar(f"Gradients/{name}_norm", grad_norm, epoch+1)
		csv_writer.writerow([epoch+1, avg_loss])
		if (epoch + 1) % 100 == 0:
			torch.save(model.state_dict(), "diffusion_policy.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")
		scheduler.step()
	torch.save(model.state_dict(), "diffusion_policy.pth")
	print("Training complete. Model saved as diffusion_policy.pth.")
	writer.close()
	csv_file.close()

if __name__ == "__main__":
	train()

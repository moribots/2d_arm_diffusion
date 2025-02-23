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
from config import *
from einops import rearrange

# ------------------------- Dataset Definition -------------------------
class PolicyDataset(Dataset):
	"""
	PolicyDataset loads training samples from JSON files stored in TRAINING_DATA_DIR.
	
	Each sample is expected to have:
	- "goal_pose": The desired pose [x, y, theta].
	- "T_pose": The current object pose. If multiple poses are provided, the last two are used.
	- "action": The end-effector (EE) position (a 2D vector) or a sequence of actions recorded during data collection.
	
	The condition is the concatenation of goal_pose, previous T_pose, and current T_pose.
	"""
	def __init__(self, data_dir):
		self.samples = []
		for filename in os.listdir(data_dir):
			if filename.endswith('.json'):
				filepath = os.path.join(data_dir, filename)
				with open(filepath, 'r') as f:
					data = json.load(f)
					self.samples.extend(data)
		# Compute normalization stats for conditions and actions.
		conditions = []
		actions_list = []
		for sample in self.samples:
			# Process condition.
			goal_pose = torch.tensor(sample["goal_pose"], dtype=torch.float32)  # (3,)
			if isinstance(sample["T_pose"][0], (list, tuple)):
				T_pose_seq = torch.tensor(sample["T_pose"], dtype=torch.float32)  # (seq_len, 3)
				if T_pose_seq.shape[0] >= 2:
					T_pose_prev = T_pose_seq[-2]
					T_pose_curr = T_pose_seq[-1]
				else:
					T_pose_prev = T_pose_seq[-1]
					T_pose_curr = T_pose_seq[-1]
			else:
				T_pose = torch.tensor(sample["T_pose"], dtype=torch.float32)  # (3,)
				T_pose_prev = T_pose
				T_pose_curr = T_pose
			cond = torch.cat([goal_pose, T_pose_prev, T_pose_curr], dim=0)  # (9,)
			conditions.append(cond)
			
			# Process action.
			action_data = sample["action"]
			if isinstance(action_data[0], (list, tuple)):
				act = torch.tensor(action_data, dtype=torch.float32)  # (seq_len, action_dim)
			else:
				act = torch.tensor(action_data, dtype=torch.float32)  # (action_dim,)
				act = act.unsqueeze(0)  # (1, action_dim)
			actions_list.append(act)
		conditions_cat = torch.stack(conditions, dim=0)  # (num_samples, 9)
		self.condition_mean = conditions_cat.mean(dim=0)
		self.condition_std = conditions_cat.std(dim=0) + 1e-6  # avoid division by zero

		actions_cat = torch.cat(actions_list, dim=0)  # (total_timesteps, action_dim)
		self.action_mean = actions_cat.mean(dim=0)
		self.action_std = actions_cat.std(dim=0) + 1e-6

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		goal_pose = torch.tensor(sample["goal_pose"], dtype=torch.float32)
		if isinstance(sample["T_pose"][0], (list, tuple)):
			T_pose_seq = torch.tensor(sample["T_pose"], dtype=torch.float32)
			if T_pose_seq.shape[0] >= 2:
				T_pose_prev = T_pose_seq[-2]
				T_pose_curr = T_pose_seq[-1]
			else:
				T_pose_prev = T_pose_seq[-1]
				T_pose_curr = T_pose_seq[-1]
		else:
			T_pose = torch.tensor(sample["T_pose"], dtype=torch.float32)
			T_pose_prev = T_pose
			T_pose_curr = T_pose
		condition = torch.cat([goal_pose, T_pose_prev, T_pose_curr], dim=0)  # (9,)
		# Normalize condition.
		condition = (condition - self.condition_mean) / self.condition_std

		action_data = sample["action"]
		if isinstance(action_data[0], (list, tuple)):
			action = torch.tensor(action_data, dtype=torch.float32)  # (seq_len, action_dim)
		else:
			action = torch.tensor(action_data, dtype=torch.float32)  # (action_dim,)
		# Normalize action.
		if action.ndim == 1:
			action = (action - self.action_mean) / self.action_std
		else:
			action = (action - self.action_mean) / self.action_std
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
	# Use a cosine annealing scheduler.
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
	mse_loss = nn.MSELoss(reduction="none")
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)
	betas = get_beta_schedule(T)
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)

	# ------------------------- Training Loop -------------------------
	for epoch in range(EPOCHS):
		running_loss = 0.0
		for condition, action in dataloader:
			condition = condition.to(device)  # (batch, 9)
			action = action.to(device)  # (batch, action_dim) or (batch, seq_len, action_dim)

			# Create a temporal window of actions from t-1 to t+WINDOW_SIZE-1.
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
			# x_t now has shape: (batch, target_length, action_dim)
			noise_pred = model(x_t, t.float(), condition)
			
			# --- Loss Weighting ---
			# Weight the loss by sqrt(1 - alpha_bar) per sample (broadcasted to the window and action dims).
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
				print(f"Gradient norm for {name}: {grad_norm:.6f}")
		csv_writer.writerow([epoch+1, avg_loss])
		# Overwrite the trained policy every 100 epochs.
		if (epoch + 1) % 100 == 0:
			torch.save(model.state_dict(), "diffusion_policy.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")
		# Step the learning rate scheduler.
		scheduler.step()
	torch.save(model.state_dict(), "diffusion_policy.pth")
	print("Training complete. Model saved as diffusion_policy.pth.")
	writer.close()
	csv_file.close()

if __name__ == "__main__":
	train()

"""
Training module for the diffusion policy.

Notes:
  - Applies a per-sample weighting factor (sqrt(1 - alpha_bar)) to the loss.
  - Optionally masks the loss for padded regions in the action sequence.
  - Restricts diffusion timesteps to a middle range (t_min to t_max) to prevent trivial targets.
"""

import os
import json
import csv  # For CSV dumps of training metrics.
import math  # For cosine annealing scheduler.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization.
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from config import *
from einops import rearrange
from normalize import Normalize  # Normalization helper class.
from PIL import Image  # For image loading and processing.
from datasets import load_dataset
import numpy as np
import wandb

def get_chunk_time_encoding(length: int):
	"""
	Returns a 1D tensor of shape (length,) that linearly scales from 0 to 1.
	This is used to embed each timestep within a chunk.
	"""
	return torch.linspace(0, 1, steps=length)

class PolicyDataset(Dataset):
	"""
	PolicyDataset loads training samples for the diffusion policy.
	
	For LeRobot, it loads the Hugging Face dataset "lerobot/pusht_image",
	groups frames by episode, and constructs examples with:
	  - observation.state: two states (t-1 and t)
	  - observation.image: two images (t-1 and t)
	  - action: sequence of actions from t-1 to t+WINDOW_SIZE.
	
	For custom data, the dataset follows a similar structure.
	"""
	def __init__(self, data_dir):
		print_settings()
		self.samples = []
		self._chunked_samples = []
		if DATASET_TYPE == "lerobot":
			self.data_source = "lerobot"
			hf_dataset = load_dataset("lerobot/pusht_image", split="train")
			print(f'Loaded {len(hf_dataset)} samples from LeRobot dataset.')
			episodes = {}
			for sample in hf_dataset:
				ep = sample["episode_index"]
				episodes.setdefault(ep, []).append(sample)
			for ep, ep_samples in episodes.items():
				ep_samples = sorted(ep_samples, key=lambda s: s["frame_index"])
				if len(ep_samples) < WINDOW_SIZE + 2:
					continue
				for i in range(1, len(ep_samples) - WINDOW_SIZE):
					obs = {
						"state": [
							ep_samples[i-1]["observation.state"],
							ep_samples[i]["observation.state"]
						],
						"image": [
							ep_samples[i-1]["observation.image"],
							ep_samples[i]["observation.image"]
						]
					}
					actions = []
					for j in range(i - 1, i + WINDOW_SIZE + 1):
						actions.append(ep_samples[j]["action"])
					new_sample = {
						"observation": obs,
						"action": actions,
						"time_index": list(range(WINDOW_SIZE + 1))
					}
					self._chunked_samples.append(new_sample)
		else:
			self.data_source = "custom"
			episodes = {}
			for filename in os.listdir(data_dir):
				if filename.endswith('.json'):
					filepath = os.path.join(data_dir, filename)
					with open(filepath, 'r') as f:
						data = json.load(f)
						for sample in data:
							ep = sample["episode_index"]
							episodes.setdefault(ep, []).append(sample)
			print(f'Loaded {len(episodes)} episodes from custom dataset.')
			for ep, ep_samples in episodes.items():
				ep_samples = sorted(ep_samples, key=lambda s: s["frame_index"])
				if len(ep_samples) < WINDOW_SIZE + 2:
					continue
				for i in range(1, len(ep_samples) - WINDOW_SIZE):
					obs = {
						"state": [
							ep_samples[i-1]["observation.state"],
							ep_samples[i]["observation.state"]
						],
						"image": [
							ep_samples[i-1]["observation.image"],
							ep_samples[i]["observation.image"]
						]
					}
					actions = []
					for j in range(i - 1, i + WINDOW_SIZE + 1):
						actions.append(ep_samples[j]["action"])
					new_sample = {
						"observation": obs,
						"action": actions,
						"time_index": list(range(WINDOW_SIZE + 1))
					}
					self._chunked_samples.append(new_sample)
		os.makedirs(OUTPUT_DIR + self.data_source, exist_ok=True)
		self.normalize = Normalize.compute_from_samples(self._chunked_samples)
		self.normalize.save(OUTPUT_DIR + self.data_source + "/" + "normalization_stats.parquet")

	def __len__(self):
		return len(self._chunked_samples)

	def __getitem__(self, idx):
		sample = self._chunked_samples[idx]
		obs = sample["observation"]
		if "state" not in obs:
			raise KeyError("Sample must contain observation['state']")
		state_raw = torch.tensor(obs["state"], dtype=torch.float32)
		state_flat = state_raw.flatten()
		condition = self.normalize.normalize_condition(state_flat)
		if self.data_source == "lerobot":
			image_array0 = obs["image"][0]
			image_array1 = obs["image"][1]
			if not isinstance(image_array0, np.ndarray):
				image_array0 = np.array(image_array0, dtype=np.uint8)
			if not isinstance(image_array1, np.ndarray):
				image_array1 = np.array(image_array1, dtype=np.uint8)
			image0 = Image.fromarray(image_array0)
			image1 = Image.fromarray(image_array1)
			image0 = image_transform(image0)
			image1 = image_transform(image1)
			image = [image0, image1]
		else:
			image_files = obs["image"]
			if len(image_files) > 1:
				img_path0 = os.path.join(self.data_source + "/" + TRAINING_DATA_DIR, "images", image_files[0])
				img_path1 = os.path.join(self.data_source + "/" + TRAINING_DATA_DIR, "images", image_files[1])
			else:
				img_path0 = os.path.join(self.data_source + "/" + TRAINING_DATA_DIR, "images", image_files[0])
				img_path1 = img_path0
			if os.path.exists(img_path0) and os.path.exists(img_path1):
				image0 = Image.open(img_path0).convert("RGB")
				image1 = Image.open(img_path1).convert("RGB")
				image0 = image_transform(image0)
				image1 = image_transform(image1)
				image = [image0, image1]
			else:
				raise FileNotFoundError("Image file not found.")
		if "action" not in sample:
			raise KeyError("Sample does not contain 'action'")
		action_data = sample["action"]
		if isinstance(action_data[0], (list, tuple)):
			action = torch.tensor(action_data, dtype=torch.float32)
		else:
			action = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
		action = self.normalize.normalize_action(action)
		if "time_index" in sample:
			time_idx = torch.tensor(sample["time_index"], dtype=torch.float32)
			max_t = float(time_idx[-1]) if time_idx[-1] > 0 else 1.0
			time_seq = time_idx / max_t
		else:
			time_seq = get_chunk_time_encoding(action.shape[0])
		return condition, image, action, time_seq

def train():
	"""
	Train the policy and log stats.

	The training loop:
	  - Loads data and prepares fixed-length action sequences with padding and masking.
	  - Samples diffusion timesteps.
	  - Computes the noise prediction loss with optional per-sample weighting factor based on the noise level.
	  - Logs training metrics (loss, learning rate, gradients) to WandB, TensorBoard, and a CSV file.
	  - Saves model checkpoints periodically.
	"""
	# Initialize a new WandB run with project settings and hyperparameters.
	wandb.init(project="diffusion_policy", config={
		"epochs": EPOCHS,
		"batch_size": BATCH_SIZE,
		"learning_rate": OPTIMIZER_LR,
		"optimizer_betas": OPTIMIZER_BETAS,
		"optimizer_eps": OPTIMIZER_EPS,
		"optimizer_weight_decay": OPTIMIZER_WEIGHT_DECAY,
		"T": T,
		"window_size": WINDOW_SIZE + 1,
		"dataset": DATASET_TYPE
	})

	# Initialize TensorBoard writer and CSV logger.
	writer = SummaryWriter(log_dir="runs/diffusion_policy_training")
	csv_file = open("training_metrics.csv", "w", newline="")
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(["epoch", "avg_loss"])

	# Enable benchmark mode for optimized performance on fixed-size inputs.
	torch.backends.cudnn.benchmark = True

	# Load data.
	dataset = PolicyDataset(TRAINING_DATA_DIR)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

	print("len dataloader", len(dataloader))
	print("len dataset", len(dataset))

	# Initialize the diffusion policy model.
	model = DiffusionPolicy(
		action_dim=ACTION_DIM,
		condition_dim=CONDITION_DIM,
		time_embed_dim=128,
		window_size=WINDOW_SIZE + 1
	)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)

	# Enable WandB to automatically log model gradients and parameters.
	wandb.watch(model, log="all")

	# Load pre-trained weights if a checkpoint exists.
	checkpoint_path = os.path.join(OUTPUT_DIR + dataset.data_source + "/", "diffusion_policy.pth")
	if os.path.exists(checkpoint_path):
		print(f"Loading pre-trained policy from {checkpoint_path}")
		state_dict = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(state_dict)

	# Use multiple GPUs if available.
	if torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs for training.")
		model = nn.DataParallel(model)

	# Set up the AdamW optimizer.
	optimizer = optim.AdamW(
		model.parameters(),
		lr=OPTIMIZER_LR,
		betas=OPTIMIZER_BETAS,
		eps=OPTIMIZER_EPS,
		weight_decay=OPTIMIZER_WEIGHT_DECAY
	)

	# Set up a cosine annealing learning rate scheduler.
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

	# Define the MSE loss (without reduction) to optionally apply custom weighting.
	mse_loss = nn.MSELoss(reduction="none")

	# Generate the beta schedule and compute the cumulative product of alphas.
	betas = get_beta_schedule(T)
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)

	# Training loop over epochs.
	for epoch in range(EPOCHS):
		running_loss = 0.0

		# Loop over batches.
		for batch in dataloader:
			# Missing key check.
			if len(batch) != 4:
				print("Malformed batch, skipping.")
				continue  # Skip any malformed batch.
			state, image, action, time_seq = batch
			state = state.to(device)
			image = [img.to(device) for img in image]
			action = action.to(device)
			time_seq = time_seq.to(device)

			# Ensure action sequence has fixed length (WINDOW_SIZE + 1).
			target_length = WINDOW_SIZE + 1
			if action.ndim == 2:  # Single action sample.
				action_seq = action.unsqueeze(1).repeat(1, target_length, 1)
				mask = torch.ones((action.size(0), target_length), dtype=torch.bool, device=device)
			elif action.ndim == 3:  # Action sequence provided.
				seq_len = action.shape[1]
				if seq_len < target_length:
					valid_mask = torch.ones((action.size(0), seq_len), dtype=torch.bool, device=device)
					pad_mask = torch.zeros((action.size(0), target_length - seq_len), dtype=torch.bool, device=device)
					mask = torch.cat([valid_mask, pad_mask], dim=1)
					# Pad the sequence with the last valid action.
					pad = action[:, -1:, :].repeat(1, target_length - seq_len, 1)
					action_seq = torch.cat([action, pad], dim=1)
				else:
					action_seq = action[:, :target_length, :]
					mask = torch.ones((action_seq.size(0), target_length), dtype=torch.bool, device=device)
			else:
				raise ValueError("Unexpected shape for action tensor")

			# Sample diffusion timesteps in the middle range to avoid trivial training targets.
			t_min = T // 10        # e.g., if T=1000, then t_min=100.
			t_max = T - t_min      # e.g., then t_max=900.
			t_tensor = torch.randint(t_min, t_max, (action_seq.size(0),), device=device)

			# Compute the noise scaling factor from the cumulative product of alphas.
			alpha_bar = rearrange(alphas_cumprod[t_tensor], 'b -> b 1 1')
			noise = torch.randn_like(action_seq)
			# Compute the noisy action sequence using the forward diffusion process:
			# x_t = sqrt(alpha_bar)*action_seq + sqrt(1 - alpha_bar)*noise.
			x_t = torch.sqrt(alpha_bar) * action_seq + torch.sqrt(1 - alpha_bar) * noise

			# Predict noise using the diffusion policy model.
			noise_pred = model(x_t, t_tensor.float(), state, image)

			# Compute the weight factor based on the noise level.
			weight = torch.sqrt(1 - alpha_bar)

			# Calculate the element-wise MSE loss.
			loss_elements = mse_loss(noise_pred, noise)

			# Apply loss masking for padded elements if enabled.
			if DO_MASK_LOSS_FOR_PADDING:
				loss_elements = loss_elements * mask.unsqueeze(-1)

			# Optionally weight the loss by sqrt(1 - alpha_bar).
			if DO_SQRT_ALPHA_BAR_WEIGHTING:
				loss = torch.mean(weight * loss_elements)
			else:
				loss = torch.mean(loss_elements)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * action_seq.size(0)

		# Compute average loss for the epoch.
		avg_loss = running_loss / len(dataset)
		current_lr = optimizer.param_groups[0]['lr']

		# Log progress to TensorBoard and CSV.
		print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
		writer.add_scalar("Loss/avg_loss", avg_loss, epoch+1)
		writer.add_scalar("LearningRate", current_lr, epoch+1)
		csv_writer.writerow([epoch+1, avg_loss])
		
		# Log metrics to WandB.
		wandb.log({"epoch": epoch+1, "avg_loss": avg_loss, "learning_rate": current_lr})

		# Log parameter histograms and gradient norms.
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch+1)
			if param.grad is not None:
				grad_norm = param.grad.data.norm(2).item()
				writer.add_scalar(f"Gradients/{name}_norm", grad_norm, epoch+1)
		
		# Save a checkpoint every 10 epochs.
		if (epoch + 1) % 10 == 0:
			torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy_final.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")
		
		# Update the learning rate scheduler.
		scheduler.step()

	# Save the final model after training completes.
	torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy.pth")
	print(f"Training complete. Model saved as {OUTPUT_DIR}diffusion_policy.pth.")
	writer.close()
	csv_file.close()
	wandb.finish()

if __name__ == "__main__":
	train()

"""
Training module for the diffusion policy.
Defines PolicyDataset that loads samples with two images (t-1 and t) along with corresponding actions.
Training metrics are logged via TensorBoard and CSV.
"""

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
from normalize import Normalize  # New normalization helper class
from PIL import Image  # For image loading and processing
from datasets import load_dataset
import numpy as np

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
	
	For custom data, the dataset now follows the same structure as the LeRobot dataset.
	It groups samples by episode_index, sorts each episode by frame_index, and then chunks
	the episode to produce examples.
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
							ep_samples[i-1]["observation.state"],  # State at t-1.
							ep_samples[i]["observation.state"]       # State at t.
						],
						"image": [
							ep_samples[i-1]["observation.image"],    # Image at t-1.
							ep_samples[i]["observation.image"]         # Image at t.
						]
					}
					actions = []
					# Collect actions from t-1 to t+WINDOW_SIZE.
					for j in range(i - 1, i + WINDOW_SIZE + 1):
						actions.append(ep_samples[j]["action"])
					new_sample = {
						"observation": obs,
						"action": actions,
						"time_index": list(range(WINDOW_SIZE + 1))
					}
					self._chunked_samples.append(new_sample)
		else:
			# Updated custom branch to mirror the lerobot branch.
			self.data_source = "custom"
			episodes = {}
			# Expect custom JSON files to have "episode_index" field.
			for filename in os.listdir(data_dir):
				if filename.endswith('.json'):
					filepath = os.path.join(data_dir, filename)
					with open(filepath, 'r') as f:
						data = json.load(f)
						for sample in data:
							ep = sample["episode_index"]
							episodes.setdefault(ep, []).append(sample)
			# Process episodes similar to the lerobot branch.
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
		# Save normalization stats using Parquet.
		self.normalize.save(OUTPUT_DIR + self.data_source + "/" + "normalization_stats.parquet")

	def __len__(self):
		return len(self._chunked_samples)

	def __getitem__(self, idx):
		sample = self._chunked_samples[idx]
		obs = sample["observation"]
		if "state" not in obs:
			raise KeyError("Sample must contain observation['state']")
		# Convert state to tensor and flatten it.
		state_raw = torch.tensor(obs["state"], dtype=torch.float32)
		state_flat = state_raw.flatten()  # Expected shape: (4,)
		condition = self.normalize.normalize_condition(state_flat)
		# Load two images.
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
		# Process actions.
		if "action" not in sample:
			raise KeyError("Sample does not contain 'action'")
		action_data = sample["action"]
		if isinstance(action_data[0], (list, tuple)):
			action = torch.tensor(action_data, dtype=torch.float32)
		else:
			action = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
		action = self.normalize.normalize_action(action)
		# Build local time sequence.
		if "time_index" in sample:
			time_idx = torch.tensor(sample["time_index"], dtype=torch.float32)
			max_t = float(time_idx[-1]) if time_idx[-1] > 0 else 1.0
			time_seq = time_idx / max_t
		else:
			time_seq = get_chunk_time_encoding(action.shape[0])
		return condition, image, action, time_seq

def train():
	"""
	Train the diffusion policy model using PolicyDataset.
	Metrics are logged via TensorBoard and CSV.
	"""
	print(f"CUDA is available: {torch.cuda.is_available()}")
	writer = SummaryWriter(log_dir="runs/diffusion_policy_training")
	csv_file = open("training_metrics.csv", "w", newline="")
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(["epoch", "avg_loss"])
	torch.backends.cudnn.benchmark = True

	dataset = PolicyDataset(TRAINING_DATA_DIR)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

	print("len dataloader", len(dataloader))
	print("len dataset", len(dataset))

	model = DiffusionPolicy(action_dim=ACTION_DIM, condition_dim=CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE+1)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)

	checkpoint_path = os.path.join(OUTPUT_DIR + dataset.data_source + "/", "diffusion_policy.pth")
	if os.path.exists(checkpoint_path):
		print(f"Loading pre-trained policy from {checkpoint_path}")
		state_dict = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(state_dict)

	if torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs for training.")
		model = nn.DataParallel(model)

	optimizer = optim.AdamW(
			model.parameters(),
			lr=OPTIMIZER_LR,
			betas=OPTIMIZER_BETAS,
			eps=OPTIMIZER_EPS,
			weight_decay=OPTIMIZER_WEIGHT_DECAY
	)

	# Cosine Annealing Scheduler
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

	mse_loss = nn.MSELoss(reduction="none")

	betas = get_beta_schedule(T)
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)

	for epoch in range(EPOCHS):
		running_loss = 0.0
		for batch in dataloader:
			if len(batch) != 4:
				continue
			state, image, action, time_seq = batch
			state = state.to(device)
			image = [img.to(device) for img in image]
			action = action.to(device)
			time_seq = time_seq.to(device)

			# Pad or slice actions to fixed length.
			target_length = WINDOW_SIZE + 1
			if action.ndim == 2:
				action_seq = action.unsqueeze(1).repeat(1, target_length, 1)
				mask = torch.ones((action.size(0), target_length), dtype=torch.bool, device=device)
			elif action.ndim == 3:
				seq_len = action.shape[1]
				if seq_len < target_length:
					valid_mask = torch.ones((action.size(0), seq_len), dtype=torch.bool, device=device)
					pad_mask = torch.zeros((action.size(0), target_length - seq_len), dtype=torch.bool, device=device)
					mask = torch.cat([valid_mask, pad_mask], dim=1)
					pad = action[:, -1:, :].repeat(1, target_length - seq_len, 1)
					action_seq = torch.cat([action, pad], dim=1)
				else:
					action_seq = action[:, :target_length, :]
					mask = torch.ones((action_seq.size(0), target_length), dtype=torch.bool, device=device)
			else:
				raise ValueError("Unexpected shape for action tensor")

			t_tensor = torch.randint(0, T, (action_seq.size(0),), device=device)
			alpha_bar = rearrange(alphas_cumprod[t_tensor], 'b -> b 1 1')
			noise = torch.randn_like(action_seq)
			x_t = torch.sqrt(alpha_bar) * action_seq + torch.sqrt(1 - alpha_bar) * noise

			noise_pred = model(x_t, t_tensor.float(), state, image)
			weight = torch.sqrt(1 - alpha_bar)
			loss_elements = mse_loss(noise_pred, noise)
			if DO_MASK_LOSS_FOR_PADDING:
				loss_elements = loss_elements * mask.unsqueeze(-1)

			loss = torch.mean(weight * loss_elements)
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
			if param.grad is not None:
				grad_norm = param.grad.data.norm(2).item()
				writer.add_scalar(f"Gradients/{name}_norm", grad_norm, epoch+1)
		csv_writer.writerow([epoch+1, avg_loss])
		if (epoch + 1) % 10 == 0:
			torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")
		scheduler.step()

	torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy.pth")
	print(f"Training complete. Model saved as {OUTPUT_DIR}diffusion_policy.pth.")
	writer.close()
	csv_file.close()

if __name__ == "__main__":
	train()

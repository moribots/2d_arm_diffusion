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

from PIL import Image  # new import for image loading
from datasets import load_dataset
import numpy as np

class PolicyDataset(Dataset):
	"""
	PolicyDataset loads training samples for the diffusion policy.
	
	In the custom case, it loads training samples from JSON files stored in TRAINING_DATA_DIR.
	In the LeRobot case (DATASET_TYPE == "lerobot"), it loads the Hugging Face dataset "lerobot/pusht",
	groups frames by episode, and constructs training examples as follows:
	  - The condition is built by concatenating:
		  - The flattened observation["state"] from frames t-1 and t (4 numbers)
		  - The image from the t-th frame is used as the visual input (extracted from the second image in a pair)
	  - The target action sequence is constructed from the actions in a temporal window 
		spanning from frame t-1 to frame t+WINDOW_SIZE.
	Normalization of conditions and actions is handled via the Normalize class.
	"""
	def __init__(self, data_dir):
		self.samples = []
		# Branch for LeRobot dataset
		if DATASET_TYPE == "lerobot":
			# Load the Hugging Face dataset "lerobot/pusht" (using the 'train' split)
			hf_dataset = load_dataset("lerobot/pusht", split="train")
			# Group samples by episode_index (episode_index is stored as a 1-element list)
			episodes = {}
			for sample in hf_dataset:
				ep = sample["episode_index"][0]  # episode_index is a list with one element
				if ep not in episodes:
					episodes[ep] = []
				episodes[ep].append(sample)
			# For each episode, sort frames by frame_index and construct training examples
			for ep, ep_samples in episodes.items():
				# Sort by frame_index (frame_index is stored as a 1-element list)
				ep_samples = sorted(ep_samples, key=lambda s: s["frame_index"][0])
				# We require at least (WINDOW_SIZE + 2) frames to form one training example:
				# one for t-1 and one for t, plus WINDOW_SIZE additional actions.
				for i in range(1, len(ep_samples) - WINDOW_SIZE):
					# Build observation: condition from frames i-1 and i.
					obs = {
						"state": [
							ep_samples[i-1]["observation.state"],  # state at t-1 (list of 2 floats)
							ep_samples[i]["observation.state"]       # state at t (list of 2 floats)
						],
						"image": [
							ep_samples[i-1]["observation.image"],    # image at t-1 (assumed as array)
							ep_samples[i]["observation.image"]         # image at t (assumed as array)
						]
					}
					# Build action sequence: actions from frame i-1 to i+WINDOW_SIZE (total WINDOW_SIZE+1 actions)
					actions = []
					for j in range(i-1, i + WINDOW_SIZE + 1):
						actions.append(ep_samples[j]["action"])  # each action is a list of 2 floats
					sample_new = {
						"observation": obs,
						"action": actions
					}
					self.samples.append(sample_new)
			self._data_source = "lerobot"
		else:
			# Original method: load JSON files from data_dir
			for filename in os.listdir(data_dir):
				if filename.endswith('.json'):
					filepath = os.path.join(data_dir, filename)
					with open(filepath, 'r') as f:
						data = json.load(f)
						self.samples.extend(data)
			self._data_source = "custom"
		# Compute normalization stats using the Normalize class.
		self.normalize = Normalize.compute_from_samples(self.samples)

		# Optionally, save the normalization stats to a file for later use.
		self.normalize.save(OUTPUT_DIR + "normalization_stats.json")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		# Build state condition from observation["state"].
		if "observation" not in sample or "state" not in sample["observation"]:
			raise KeyError("Sample must contain observation['state']")
		state = torch.tensor(sample["observation"]["state"], dtype=torch.float32)  # shape (2,2) if two states
		state = state.flatten()  # (4,)
		state = self.normalize.normalize_condition(state)
		
		# Load image for visual input.
		# For the LeRobot dataset, we assume "observation.image" contains image arrays.
		if self._data_source == "lerobot":
			# Use the second image (corresponding to t) as visual input.
			image_array = sample["observation"]["image"][1]
			# If the image is not a numpy array, convert it.
			if not isinstance(image_array, np.ndarray):
				image_array = np.array(image_array, dtype=np.uint8)
			# Convert numpy array to PIL Image.
			image = Image.fromarray(image_array)
			image = image_transform(image)  # shape (3, IMG_RES, IMG_RES)
		else:
			# Custom case: "observation.image" contains image file names.
			image_files = sample["observation"]["image"]
			img_path = os.path.join(TRAINING_DATA_DIR, "images", image_files[1])
			image = Image.open(img_path).convert("RGB")
			image = image_transform(image)
		
		# Process action sequence.
		if "action" not in sample:
			raise KeyError("Sample does not contain 'action'")
		action_data = sample["action"]
		if isinstance(action_data[0], (list, tuple)):
			action = torch.tensor(action_data, dtype=torch.float32)  # (seq_len, action_dim)
		else:
			action = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
		action = self.normalize.normalize_action(action)
		
		# Optionally, return action_is_pad if present.
		if "action_is_pad" in sample:
			action_is_pad = torch.tensor(sample["action_is_pad"], dtype=torch.bool)
			return state, image, action, action_is_pad
		else:
			return state, image, action

def train():
	print(f"CUDA is available: {torch.cuda.is_available()}")
	writer = SummaryWriter(log_dir="runs/diffusion_policy_training")
	csv_file = open("training_metrics.csv", "w", newline="")
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(["epoch", "avg_loss"])
	torch.backends.cudnn.benchmark = True
	dataset = PolicyDataset(TRAINING_DATA_DIR)
	dataloader = DataLoader(
		dataset, 
		batch_size=BATCH_SIZE, 
		shuffle=True, 
		num_workers=4,
		pin_memory=True
	)
	print("len dataloader", len(dataloader))
	print("len dataset", len(dataset))
	model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE+1)
	
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	# Move model to device
	model.to(device)

	# Check if a pre-trained policy exists and load it.
	checkpoint_path = os.path.join(OUTPUT_DIR, "diffusion_policy.pth")
	if os.path.exists(checkpoint_path):
		print(f"Loading pre-trained policy from {checkpoint_path}")
		state_dict = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(state_dict)

	# Wrap the model in DataParallel if more than one GPU is available.
	if torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs for training.")
		model = nn.DataParallel(model)
	
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
	mse_loss = nn.MSELoss(reduction="none")
	
	betas = get_beta_schedule(T)
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)

	for epoch in range(EPOCHS):
		running_loss = 0.0
		for batch in dataloader:
			# Unpack batch. If action_is_pad is provided, batch has 4 elements.
			if len(batch) == 4:
				state, image, action, action_is_pad = batch
				has_mask = True
			else:
				state, image, action = batch
				has_mask = False

			state = state.to(device)  # (batch, 4)
			image = image.to(device)  # (batch, 3, IMG_RES, IMG_RES)
			action = action.to(device)
			target_length = WINDOW_SIZE + 1
			if action.ndim == 2:
				action_seq = action.unsqueeze(1).repeat(1, target_length, 1)
			elif action.ndim == 3:
				seq_len = action.shape[1]
				if seq_len < target_length:
					pad = action[:, -1:, :].repeat(1, target_length - seq_len, 1)
					action_seq = torch.cat([action, pad], dim=1)
				else:
					action_seq = action[:, :target_length, :]
			else:
				raise ValueError("Unexpected shape for action tensor")

			t = torch.randint(0, T, (action_seq.size(0),), device=device)
			alpha_bar = rearrange(alphas_cumprod[t], 'b -> b 1 1')
			noise = torch.randn_like(action_seq)
			x_t = torch.sqrt(alpha_bar) * action_seq + torch.sqrt(1 - alpha_bar) * noise

			# Forward pass: predict noise.
			noise_pred = model(x_t, t.float(), state, image)
			weight = torch.sqrt(1 - alpha_bar)

			# Always predict noise; target is noise.
			target = noise

			# Compute MSE loss between the prediction and target.
			loss_elements = mse_loss(noise_pred, target)

			# Optionally, mask loss for padded actions if provided.
			if DO_MASK_LOSS_FOR_PADDING and has_mask:
				action_is_pad = action_is_pad.to(device)
				in_episode_bound = ~action_is_pad  # Boolean mask for valid actions.
				loss_elements = loss_elements * in_episode_bound.unsqueeze(-1)

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
		for name, param in model.named_parameters():
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

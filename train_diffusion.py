"""
Training module for the diffusion policy.

Notes:
  - Applies a per-sample weighting factor (sqrt(1 - alpha_bar)) to the loss.
  - Optionally masks the loss for padded regions in the action sequence.
  - Restricts diffusion timesteps to a middle range (t_min to t_max) to prevent trivial targets.
"""

import os
import json
import math  # For cosine annealing scheduler.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from config import *
from einops import rearrange
from normalize import Normalize  # Normalization helper class.
from PIL import Image  # For image loading and processing.
from policy_inference import DiffusionPolicyInference
from datasets import load_dataset
import numpy as np
import wandb
try:
	from kaggle_secrets import UserSecretsClient
except ImportError:
	pass  # Not running on Kaggle
import gymnasium as gym
import gym_pusht
import cv2
import time  # Should already be imported
from tqdm import tqdm  # Add this import at the top
from video_utils import save_video  # Import the new video utility

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
		self.normalize = Normalize.compute_from_limits()
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

def validate_policy(model, device, save_locally=False, local_save_path=None):
	"""
	Run a validation episode using the current diffusion policy model,
	record a video, compute the total reward, and log the video and reward to WandB.
	
	Args:
		model (nn.Module): Current diffusion policy model.
		device (torch.device): Device on which the model is running.
		save_locally (bool): If True, save the video locally regardless of WandB upload.
		local_save_path (str): Optional explicit path to save the video locally. If None and save_locally=True,
							   a default path will be used.

	Returns:
		tuple: (total_reward (float), video_path (str or None)) - Total reward accumulated during validation 
			   and path to saved video (if successful and save_locally=True, else None).
	"""
	# Create the environment using the LeRobot gym environment.
	env = gym.make(LE_ROBOT_GYM_ENV_NAME, obs_type="pixels_agent_pos", render_mode="rgb_array")
	obs, info = env.reset()
	frames = []
	total_reward = 0.0
	done = False
	steps = 0
	max_steps = 100  # You can adjust the number of validation steps

	prev_agent_pos = None
	prev_image = None
	
	# Track action sequence and current position in sequence
	action_buffer = []
	current_action_idx = 0

	# Create an instance of the inference wrapper using current model weights.
	# We reuse DiffusionPolicyInference but override its model with our current model.
	inference = DiffusionPolicyInference(model_path="", T=T, device=device, norm_stats_path=OUTPUT_DIR + "lerobot/normalization_stats.parquet")
	inference.model = model  # Use current model weights

	# Add tqdm progress bar
	pbar = tqdm(total=max_steps, desc="Validation", leave=True)
	try:
		while not done and steps < max_steps:
			 # Check if we need to resample actions (empty buffer or used half of window)
			if len(action_buffer) == 0 or current_action_idx >= WINDOW_SIZE // 2:
				# Build state from current and previous agent positions.
				agent_pos_np = np.array(obs["agent_pos"])
				agent_pos_tensor = torch.from_numpy(agent_pos_np).float()
				if prev_agent_pos is None:
					state = torch.cat([agent_pos_tensor, agent_pos_tensor], dim=0).unsqueeze(0)
				else:
					state = torch.cat([prev_agent_pos.clone(), agent_pos_tensor], dim=0).unsqueeze(0)
				prev_agent_pos = agent_pos_tensor.clone()
				
				# Process current image: add batch dimension.
				image_array = obs["pixels"]
				if not isinstance(image_array, np.ndarray):
					image_array = np.array(image_array, dtype=np.uint8)
				current_image_tensor = image_transform(Image.fromarray(image_array)).unsqueeze(0)
				
				# Use previous image if available; otherwise, duplicate.
				if prev_image is None:
					image_tuple = [current_image_tensor.to(device), current_image_tensor.to(device)]
				else:
					image_tuple = [prev_image.to(device), current_image_tensor.to(device)]
				prev_image = current_image_tensor.clone()
				
				# Sample a full window of actions
				action_seq = inference.sample_action(state.to(device), image_tuple)
				
				# Store the sequence in our buffer
				action_buffer = action_seq.cpu().numpy()
				current_action_idx = 0
			
			# Use the next action from our buffer
			action = action_buffer[current_action_idx]
			current_action_idx += 1

			# Step the environment.
			obs, reward, done, truncated, info = env.step(action)
			total_reward += reward
			frame = env.render()
			if frame is not None:
				frames.append(frame)
			steps += 1
			
			# Update progress bar with current reward
			pbar.set_postfix(reward=f"{total_reward:.2f}")
			pbar.update(1)
	finally:
		# Make sure to close the progress bar even if there's an error
		pbar.close()

	env.close()

	if len(frames) == 0:
		print("No frames captured during validation.")
		return total_reward, None
	
	print(f'Validation total reward: {total_reward}, done: {done}, steps: {steps}')

	# Determine base directory for video
	video_dir = os.path.join(OUTPUT_DIR, "videos")
	os.makedirs(video_dir, exist_ok=True)
	
	# Determine video identifier (wandb step or timestamp)
	has_wandb_run = 'wandb' in globals() and wandb.run is not None
	video_identifier = f"{wandb.run.step}" if has_wandb_run else f"{int(time.time())}"
	
	# Handle local saving if requested
	if save_locally:
		if local_save_path is None:
			local_dir = os.path.join(OUTPUT_DIR, "local_videos")
			os.makedirs(local_dir, exist_ok=True)
			local_save_path = os.path.join(local_dir, f"validation_video_local_{int(time.time())}.gif")
		
		# Save video locally without WandB logging
		local_video_path, local_success = save_video(
			frames,
			os.path.dirname(local_save_path),
			os.path.basename(local_save_path).split('.')[0],  # Use basename without extension as identifier
			wandb_log=False,
			use_gif=True,  # Use GIF format for better compatibility
			save_locally=True  # Explicitly save locally
		)
		
		if not local_success:
			print(f"Failed to save video locally to {local_save_path}")
	
	# Save/log the validation video with WandB integration
	video_path, success = save_video(
		frames,
		video_dir,
		video_identifier,
		wandb_log=has_wandb_run,
		wandb_key="validation_video",
		additional_wandb_data={"validation_total_reward": total_reward},
		use_gif=True,  # Use GIF format for better compatibility
		save_locally=save_locally  # Only save locally if explicitly requested
	)
	
	# Return the appropriate video path based on what was requested
	if save_locally and 'local_video_path' in locals() and local_video_path:
		return total_reward, local_video_path
	
	return total_reward, video_path

def train():
	"""
	Train the policy and log stats.

	The training loop:
	  - Loads data and prepares fixed-length action sequences with padding and masking.
	  - Samples diffusion timesteps.
	  - Computes the noise prediction loss with optional per-sample weighting factor based on the noise level.
	  - Logs training metrics (loss, learning rate, gradients) to WandB.
	  - Saves model checkpoints periodically.
	"""
	# Retrieve the WandB API key from the environment variable
	secret_label = "WANDB_API_KEY"
	try:
		api_key = UserSecretsClient().get_secret(secret_label)
	except KeyError:
		api_key = None
	
	if api_key is None:
		print("WANDB_API_KEY is not set. Please add it as a Kaggle secret.")
		KeyError("WANDB_API_KEY is not set. Please add it as a Kaggle secret or manually.")
	else:
		# Log in to WandB using the private API key
		wandb.login(key=api_key)
	# Initialize a new WandB run with project settings and hyperparameters.
	wandb.init(entity="moribots-personal", project="2d_arm_diffusion", config={
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

	# Enable benchmark mode for optimized performance on fixed-size inputs.
	torch.backends.cudnn.benchmark = True

	# Load data.
	dataset = PolicyDataset(TRAINING_DATA_DIR)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()-1, pin_memory=True)

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
	
	# Add temporal smoothness loss
	def temporal_smoothness_loss(action_sequence):
		"""Calculate temporal smoothness loss as the mean squared difference between consecutive actions"""
		if action_sequence.shape[1] <= 1:
			return torch.tensor(0.0, device=action_sequence.device)
		diffs = action_sequence[:, 1:] - action_sequence[:, :-1]
		return torch.mean(torch.sum(diffs**2, dim=-1))
	
	# Helper function to compute total gradient norm
	def compute_grad_norm(parameters):
		"""Compute the L2 norm of gradients for all parameters"""
		total_norm = 0.0
		parameters = list(filter(lambda p: p.grad is not None, parameters))
		for p in parameters:
			param_norm = p.grad.detach().norm(2)
			total_norm += param_norm.item() ** 2
		total_norm = total_norm ** 0.5
		return total_norm

	# Generate the beta schedule and compute the cumulative product of alphas.
	betas = get_beta_schedule(T)
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)

	# Training loop over epochs.
	for epoch in range(EPOCHS):
		running_loss = 0.0
		running_smoothness_loss = 0.0
		running_grad_norm = 0.0
		batch_count = 0

		# Loop over batches.
		for batch_idx, batch in enumerate(dataloader):
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

			t_min = 0
			t_max = T - t_min
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
			
			# Add temporal smoothness loss to promote smoother actions
			# This encourages temporally coherent action sequences
			smooth_weight = 0.1  # Weight for the smoothness loss term
			if epoch >= 10:  # Start applying smoothness loss after 10 epochs
				smooth_loss = temporal_smoothness_loss(action_seq) * smooth_weight
				total_loss = loss + smooth_loss
				running_smoothness_loss += smooth_loss.item()
			else:
				total_loss = loss
			
			optimizer.zero_grad()
			total_loss.backward()
			
			 # Calculate gradient norm before clipping
			grad_norm = compute_grad_norm(model.parameters())
			running_grad_norm += grad_norm
			
			# Gradient clipping to prevent exploding gradients
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			
			optimizer.step()

			running_loss += loss.item() * action_seq.size(0)
			batch_count += 1

			# Add batch-level logging (optional)
			wandb.log({
				"batch_loss": loss.item(),
				"smoothness_loss": smooth_loss.item() if epoch >= 10 else 0.0,
				"batch_grad_norm": grad_norm,  # Log per-batch gradient norm
				"global_step": epoch * len(dataloader) + batch_idx
			})

		# Compute average loss for the epoch.
		avg_loss = running_loss / len(dataset)
		avg_smoothness_loss = running_smoothness_loss / len(dataloader) if epoch >= 10 else 0.0
		avg_grad_norm = running_grad_norm / batch_count if batch_count > 0 else 0.0
		current_lr = optimizer.param_groups[0]['lr']

		# Log progress to console
		print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}, Smoothness Loss: {avg_smoothness_loss:.6f}, Grad Norm: {avg_grad_norm:.6f}")
		
		# Log metrics to WandB.
		wandb.log({
			"epoch": epoch+1, 
			"avg_loss": avg_loss, 
			"avg_smoothness_loss": avg_smoothness_loss,
			"avg_grad_norm": avg_grad_norm,  # Log epoch average gradient norm
			"learning_rate": current_lr
		})

		if (epoch + 1) % VALIDATION_INTERVAL == 0:
			val_reward, _ = validate_policy(model, device)  # Ignore the video path
			print(f"Validation total reward at epoch {epoch+1}: {val_reward}")
		
		# Save a checkpoint every 10 epochs.
		if (epoch + 1) % 10 == 0:
			torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy_final.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")
		
		# Update the learning rate scheduler.
		scheduler.step()

	# Save the final model after training completes.
	torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy.pth")
	print(f"Training complete. Model saved as {OUTPUT_DIR}diffusion_policy.pth.")
	wandb.finish()

if __name__ == "__main__":
	train()

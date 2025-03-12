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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
		time_index = torch.tensor(sample["time_index"], dtype=torch.float32)
		max_t = float(time_index[-1])
		time_seq = time_index / max_t
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
		tuple: (total_reward (float), video_path (str or None), action_plot_path (str or None)) -
			   Total reward accumulated during validation, path to saved video (if successful),
			   and path to action plot (if successful).
	"""
	# Create the environment using the LeRobot gym environment.
	env = gym.make(LE_ROBOT_GYM_ENV_NAME, obs_type="pixels_agent_pos", render_mode="rgb_array")
	obs, info = env.reset()
	frames = []
	total_reward = 0.0
	done = False
	fps = env.metadata['render_fps']
	print(f'Sim FPS {fps}')
	max_steps = 150

	# Track previous observations for conditioning
	prev_agent_pos = None
	prev_image = None

	# Create an instance of the inference wrapper using current model weights
	inference = DiffusionPolicyInference(model_path="", T=T, device=device, norm_stats_path=OUTPUT_DIR + "lerobot/normalization_stats.parquet")
	inference.model = model  # Use current model weights

	# Track simulation time for proper action interpolation
	start_time = 0.0
	current_time = start_time

	# Initialize lists to track actions, timestamps, and new inferences
	action_history = []
	timestep_history = []
	inference_points = []  # New list to track when inferences occur

	steps = 0

	# Add tqdm progress bar
	pbar = tqdm(total=max_steps, desc="Validation", leave=True)
	try:
		while not done and steps < max_steps:
			# Build state from current and previous agent positions
			agent_pos_np = np.array(obs["agent_pos"])
			agent_pos_tensor = torch.from_numpy(agent_pos_np).float()
			if prev_agent_pos is None:
				state = torch.cat([agent_pos_tensor, agent_pos_tensor], dim=0).unsqueeze(0)
			else:
				state = torch.cat([prev_agent_pos.clone(), agent_pos_tensor], dim=0).unsqueeze(0)
			prev_agent_pos = agent_pos_tensor.clone()

			# Process current image: add batch dimension
			image_array = obs["pixels"]
			if not isinstance(image_array, np.ndarray):
				image_array = np.array(image_array, dtype=np.uint8)
			current_image_tensor = image_transform(Image.fromarray(image_array)).unsqueeze(0)

			# Use previous image if available; otherwise, duplicate
			if prev_image is None:
				image_tuple = [current_image_tensor.to(device), current_image_tensor.to(device)]
			else:
				image_tuple = [prev_image.to(device), current_image_tensor.to(device)]
			prev_image = current_image_tensor.clone()

			# Get action from policy, providing current time for proper interpolation
			# The inference class now returns whether this was a new inference
			action, is_new_inference = inference.sample_action(
				state.to(device),
				image_tuple,
				smoothing=False
			)

			 # Record action for plotting
			action_history.append(action.cpu().numpy())
			timestep_history.append(current_time)

			 # Record if this was a new inference point
			if is_new_inference:
				inference_points.append(current_time)

			# Step the environment
			obs, reward, done, truncated, info = env.step(action.cpu().numpy())
			total_reward += reward

			frame = env.render()
			if frame is not None:
				frames.append(frame)

			# Update time for next step (simulate real time passing)
			current_time = start_time + (steps + 1) * (1.0 / fps)
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
		return total_reward, None, None

	print(f'Validation total reward: {total_reward}, done: {done}, steps: {steps}')

	# Generate action plots
	action_plot_path = None
	if len(action_history) > 0:
		# Pass inference points to the plotting function
		action_plot_path = plot_actions(action_history, timestep_history, inference_points)

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

	# Compare with training data sample
	compare_to_training_data(model, device)

	# Return the appropriate video path based on what was requested
	if save_locally and 'local_video_path' in locals() and local_video_path:
		return total_reward, local_video_path, action_plot_path

	return total_reward, video_path, action_plot_path

def compare_to_training_data(model, device):
	"""
	Selects a random training data episode, compares the policy's predicted actions
	to the ground truth actions across the entire episode, computes MSE, and logs to WandB.

	Args:
		model (nn.Module): Current diffusion policy model.
		device (torch.device): Device on which the model is running.

	Returns:
		float: Mean squared error between predicted and ground truth actions.
	"""
	try:
		print("Comparing policy predictions to training data...")

		# Load training dataset
		dataset = PolicyDataset(TRAINING_DATA_DIR)

		 # Initialize lists to store all ground truth and predicted actions, and inference points
		all_pred_actions = []
		all_gt_actions = []
		all_timesteps = []
		all_inference_points = []  # New list to track inference points

		# Find all samples from a single episode
		if DATASET_TYPE == "lerobot":
			# For LeRobot dataset, directly access the HuggingFace dataset
			# to get a complete episode
			try:
				import random
				hf_dataset = load_dataset("lerobot/pusht_image", split="train")

				# Get unique episode indices
				episode_indices = set()
				for sample in hf_dataset:
					episode_indices.add(sample["episode_index"])

				if not episode_indices:
					raise ValueError("No episodes found in dataset")

				# Select a random episode
				selected_ep = random.choice(list(episode_indices))
				print(f"Selected episode {selected_ep} for comparison")

				# Get all frames from this episode
				episode_frames = []
				for sample in hf_dataset:
					if sample["episode_index"] == selected_ep:
						episode_frames.append(sample)

				# Sort by frame index
				episode_frames.sort(key=lambda x: x["frame_index"])
				print(f"Found {len(episode_frames)} frames for episode {selected_ep}")

				# Process a significantly longer sequence of frames
				max_samples = min(500, len(episode_frames) - 1)  # Increased from 100 to 500

				 # Create inference wrapper with current model
				inference = DiffusionPolicyInference(model_path="", T=T, device=device,
										   norm_stats_path=OUTPUT_DIR + "lerobot/normalization_stats.parquet")
				inference.model = model

				# Reset inference buffers
				inference.action_buffer = []
				inference.current_action_idx = 0

				# Track simulated time for action generation
				current_time = 0.0

				# Process each frame in the episode (skipping the first one)
				for i in range(1, max_samples):
					# Create condition from current and previous frame
					prev_state = torch.tensor(episode_frames[i-1]["observation.state"], dtype=torch.float32)
					curr_state = torch.tensor(episode_frames[i]["observation.state"], dtype=torch.float32)

					# Flatten and concatenate states
					condition = torch.cat([prev_state.flatten(), curr_state.flatten()], dim=0)
					condition = dataset.normalize.normalize_condition(condition)
					condition = condition.unsqueeze(0).to(device)

					# Process images
					prev_img = episode_frames[i-1]["observation.image"]
					curr_img = episode_frames[i]["observation.image"]

					if not isinstance(prev_img, np.ndarray):
						prev_img = np.array(prev_img, dtype=np.uint8)
					if not isinstance(curr_img, np.ndarray):
						curr_img = np.array(curr_img, dtype=np.uint8)

					prev_img_tensor = image_transform(Image.fromarray(prev_img)).unsqueeze(0).to(device)
					curr_img_tensor = image_transform(Image.fromarray(curr_img)).unsqueeze(0).to(device)

					image_tuple = [prev_img_tensor, curr_img_tensor]

					# Get ground truth action (unnormalized)
					gt_action = torch.tensor(episode_frames[i]["action"], dtype=torch.float32).to(device)

					# Get predicted action, now also track new inferences
					pred_action, is_new_inference = inference.sample_action(
						condition,
						image_tuple,
						smoothing=False
					)

					# Store for analysis
					all_pred_actions.append(pred_action.cpu().numpy())
					all_gt_actions.append(gt_action.cpu().numpy())
					all_timesteps.append(current_time)

					# Track if this was a new inference
					if is_new_inference:
						all_inference_points.append(current_time)

					# Increment time
					current_time += SEC_PER_SAMPLE

					# Print progress occasionally
					if i % 50 == 0:
						print(f"Processed {i}/{max_samples} frames, current time: {current_time:.2f}s")

			except Exception as e:
				print(f"Error accessing HuggingFace dataset directly: {e}")
				print("Falling back to dataset samples")

				# Fallback to the original method
				episode_indices = set()
				for sample in dataset._chunked_samples:
					ep_idx = sample.get("episode_index", None)
					if ep_idx is None and "observation" in sample:
						# Try to extract from the first action if available
						if "action" in sample and len(sample["action"]) > 0:
							if isinstance(sample["action"][0], dict):
								ep_idx = sample["action"][0].get("episode_index", None)

					if ep_idx is not None:
						episode_indices.add(ep_idx)

				if not episode_indices:
					# Fallback - use a random sample and its window only
					print("Warning: Could not identify complete episodes, using single sample")
					sample_idx = np.random.randint(0, len(dataset))
					episode_samples = [sample_idx]
				else:
					# Select a random episode
					selected_ep = random.choice(list(episode_indices))
					# Find all samples from this episode
					episode_samples = []
					for i, sample in enumerate(dataset._chunked_samples):
						ep_idx = sample.get("episode_index", None)
						if ep_idx is None and "observation" in sample:
							if "action" in sample and len(sample["action"]) > 0:
								if isinstance(sample["action"][0], dict):
									ep_idx = sample["action"][0].get("episode_index", None)

						if ep_idx == selected_ep:
							episode_samples.append(i)

					# Sort by frame index if available
					if episode_samples:
						try:
							episode_samples.sort(key=lambda i: dataset._chunked_samples[i].get("frame_index", i))
						except (KeyError, TypeError):
							# If sorting fails, keep original order
							pass

						print(f"Selected episode {selected_ep} with {len(episode_samples)} samples")
					else:
						# Fallback to a random sample
						print(f"Warning: No samples found for episode {selected_ep}, using random sample")
						sample_idx = np.random.randint(0, len(dataset))
						episode_samples = [sample_idx]
		else:
			# For custom datasets, randomly select an episode
			sample_idx = np.random.randint(0, len(dataset))
			episode_samples = [sample_idx]
			print("Using a single random sample for custom dataset")

			# Process each sample in the episode
			for t, sample_idx in enumerate(episode_samples):
				condition, image_tuple, normalized_gt_actions, time_seq = dataset[sample_idx]

				# Unnormalize ground truth actions for fair comparison with predictions
				# (inference class will unnormalize its output)
				gt_actions = dataset.normalize.unnormalize_action(normalized_gt_actions).to(device)

				# Create inference wrapper with current model
				inference = DiffusionPolicyInference(model_path="", T=T, device=device,
									   norm_stats_path=OUTPUT_DIR + "lerobot/normalization_stats.parquet")
				inference.model = model

				# Reset inference buffers
				inference.action_buffer = []
				inference.current_action_idx = 0

				# Track simulated time for action generation
				current_time = 0.0

				# Skip the first action (t-1) as it's just for conditioning
				# Only take actions from t=1 onwards
				for i in range(1, len(gt_actions)):
					# Get the state and image data for this timestep
					current_condition = condition.unsqueeze(0).to(device)
					current_image_tuple = [img.unsqueeze(0).to(device) for img in image_tuple]

					# Get predicted action at this timestep
					action = inference.sample_action(
						current_condition,
						current_image_tuple,
						smoothing=False
					)

					# Record prediction, ground truth, and timestep
					all_pred_actions.append(action.cpu().numpy())
					all_gt_actions.append(gt_actions[i].cpu().numpy())
					all_timesteps.append(current_time)

					# Move forward in simulated time
					current_time += SEC_PER_SAMPLE

					# Limit to a reasonable number of samples if episode is very long
					if len(all_timesteps) >= 100:  # Set a reasonable maximum
						print(f"Limiting comparison to {len(all_timesteps)} samples")
						break

		# Handle case where we couldn't get any valid data
		if not all_pred_actions:
			print("Warning: No valid actions found for comparison")
			return float('inf')

		# Convert lists to numpy arrays for easier manipulation
		pred_actions = np.array(all_pred_actions)
		gt_actions_np = np.array(all_gt_actions)
		timesteps = np.array(all_timesteps)

		# Compute MSE between predictions and ground truth
		mse = np.mean((pred_actions - gt_actions_np) ** 2)
		print(f"Training data comparison MSE: {mse:.6f} over {len(timesteps)} timesteps ({timesteps[-1]:.2f}s)")

		 # Create comparison plot
		fig = make_subplots(
			rows=ACTION_DIM,
			cols=1,
			shared_xaxes=True,
			subplot_titles=[f'Action Dimension {i}' for i in range(ACTION_DIM)]
		)

		colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

		# Plot each action dimension separately
		for dim in range(ACTION_DIM):
			color_idx = dim % len(colors)

			# Add predicted actions
			fig.add_trace(
				go.Scatter(
					x=timesteps,
					y=pred_actions[:, dim],
					mode='lines+markers',
					name=f'Predicted Dim {dim}',
					line=dict(color=colors[color_idx], width=2, dash='solid'),
					marker=dict(symbol='circle', size=8),
					hovertemplate='Time: %{x:.3f}<br>Predicted: %{y:.3f}'
				),
				row=dim+1,
				col=1
			)

			# Add ground truth actions
			fig.add_trace(
				go.Scatter(
					x=timesteps,
					y=gt_actions_np[:, dim],
					mode='lines+markers',
					name=f'Ground Truth Dim {dim}',
					line=dict(color='black', width=1, dash='dash'),
					marker=dict(symbol='x', size=8),
					hovertemplate='Time: %{x:.3f}<br>Ground Truth: %{y:.3f}'
				),
				row=dim+1,
				col=1
			)

			# Add error bars/regions to show difference between predicted and ground truth
			error = np.abs(pred_actions[:, dim] - gt_actions_np[:, dim])
			fig.add_trace(
				go.Scatter(
					x=timesteps,
					y=error,
					mode='lines',
					name=f'Error Dim {dim}',
					line=dict(color='rgba(255,0,0,0.3)', width=0),
					fill='tozeroy',
					fillcolor='rgba(255,0,0,0.2)',
					hovertemplate='Time: %{x:.3f}<br>Error: %{y:.3f}'
				),
				row=dim+1,
				col=1
			)

		# Update layout
		fig.update_layout(
			title=f"Predicted vs Ground Truth Actions (MSE: {mse:.6f})",
			xaxis_title="Time (seconds)",
			yaxis_title="Action Value",
			legend_title="Action Type",
			width=1000,
			height=300 * ACTION_DIM,
			hovermode="closest"
		)

		# Add red dotted vertical lines to mark new inference points
		if all_inference_points and len(all_inference_points) > 0:
			for t in all_inference_points:
				fig.add_vline(
					x=t,
					line=dict(color="purple", width=2, dash="dot"),
					annotation_text="inference",
					annotation_position="top left",
					annotation=dict(
						font=dict(color="purple", size=10),
						bordercolor="purple",
						borderwidth=1,
						bgcolor="rgba(255,255,255,0.8)"
					)
				)

		# Check if WandB is active
		has_wandb_run = 'wandb' in globals() and wandb.run is not None

		# Create paths for potential saving
		comparison_dir = os.path.join(OUTPUT_DIR, "train_comparisons")
		timestamp = int(time.time())
		html_path = os.path.join(comparison_dir, f"train_comparison_{timestamp}.html")
		png_path = os.path.join(comparison_dir, f"train_comparison_{timestamp}.png")

		# Save files only if WandB is not active
		if not has_wandb_run:
			# Ensure output directory exists
			os.makedirs(comparison_dir, exist_ok=True)

			# Save interactive HTML and static PNG
			fig.write_html(html_path)
			fig.write_image(png_path)
			print(f"Comparison plots saved to {html_path} (interactive) and {png_path} (static)")
		else:
			# For WandB, we need to temporarily save the PNG for logging
			os.makedirs(comparison_dir, exist_ok=True)
			temp_png_path = os.path.join(comparison_dir, f"temp_comparison_{timestamp}.png")
			fig.write_image(temp_png_path)

			# Log to WandB
			wandb.log({
				"train_comparison_mse": mse,
				"train_comparison_plot": wandb.Image(temp_png_path),
				"train_comparison_data": wandb.Table(
					data=[
						[t, *pred, *gt, *np.abs(np.array(pred) - np.array(gt))]
						for t, pred, gt in zip(timesteps, pred_actions, gt_actions_np)
					],
					columns=["timestep"] +
							[f"pred_dim_{i}" for i in range(ACTION_DIM)] +
							[f"gt_dim_{i}" for i in range(ACTION_DIM)] +
							[f"error_dim_{i}" for i in range(ACTION_DIM)]
				)
			})

			# Remove temporary file after WandB has uploaded it
			try:
				os.remove(temp_png_path)
			except:
				pass  # Ignore if file removal fails

		return mse

	except Exception as e:
		print(f"Error in training data comparison: {e}")
		import traceback
		traceback.print_exc()
		return float('inf')  # Return infinity on error

def plot_actions(action_history, timestep_history, inference_points=None):
	"""
	Plot each dimension of the actions over time using Plotly and save the plot.

	Marks the beginning of each policy inference window to visualize when
	new actions were sampled from the diffusion model. Adds vertical red dotted
	lines at moments when a new inference was performed.

	Args:
		action_history (list): List of action arrays from each timestep
		timestep_history (list): List of timestep values corresponding to each action
		inference_points (list): Optional list of timesteps when new inferences were performed

	Returns:
		str: Path to the saved plot or None if saving failed
	"""
	try:
		# Convert lists to numpy arrays for easier handling
		actions = np.array(action_history)
		timesteps = np.array(timestep_history)

		# Get the number of action dimensions
		action_dim = actions.shape[1]

		# Create plotly figure
		fig = go.Figure()

		# Color palette for action dimensions
		colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
				  'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
				  'rgb(148, 103, 189)', 'rgb(140, 86, 75)']

		# Plot each action dimension
		for dim in range(action_dim):
			color_idx = dim % len(colors)
			fig.add_trace(go.Scatter(
				x=timesteps,
				y=actions[:, dim],
				mode='lines',
				name=f'Action Dim {dim}',
				line=dict(color=colors[color_idx], width=2),
				hovertemplate='Time: %{x:.3f}<br>Value: %{y:.3f}'
			))

		# Add red dotted vertical lines to highlight when new inferences occurred
		if inference_points and len(inference_points) > 0:
			for t in inference_points:
				fig.add_vline(
					x=t,
					line=dict(color="purple", width=3, dash="dot")
				)

		# Update layout with improved formatting
		fig.update_layout(
			title="Action Dimensions Over Time",
			xaxis_title="Timestep",
			yaxis_title="Action Value",
			legend_title="Dimensions",
			font=dict(family="Arial, sans-serif", size=12),
			hovermode="closest",
			plot_bgcolor='rgba(250,250,250,0.9)',
			width=1000,
			height=600,
			margin=dict(l=60, r=30, t=60, b=60)
		)

		# Add grid
		fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
		fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')

		# Ensure output directory exists
		plot_dir = os.path.join(OUTPUT_DIR, "action_plots")
		os.makedirs(plot_dir, exist_ok=True)

		# Generate unique filenames with timestamp
		timestamp = int(time.time())
		html_path = os.path.join(plot_dir, f"action_plot_{timestamp}.html")
		png_path = os.path.join(plot_dir, f"action_plot_{timestamp}.png")

		# Save the plot as interactive HTML
		fig.write_html(html_path)

		# Save as static image for WandB and other displays
		fig.write_image(png_path)

		# Log to WandB if available
		if 'wandb' in globals() and wandb.run is not None:
			wandb.log({
				"action_plot": wandb.Image(png_path),
				"action_data": wandb.Table(
					data=[[t, *a] for t, a in zip(timesteps, actions)],
					columns=["timestep"] + [f"action_dim_{i}" for i in range(action_dim)]
				)
			})
			# Also log the HTML file as an artifact
			artifact = wandb.Artifact(f"action_plot_{timestamp}", type="plot")
			artifact.add_file(html_path)
			wandb.log_artifact(artifact)

		print(f"Action plot saved to {html_path} (interactive) and {png_path} (static)")
		return png_path  # Return the path to the PNG for consistency with previous code

	except Exception as e:
		print(f"Failed to create action plot: {e}")
		import traceback
		traceback.print_exc()
		return None

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
	except NameError:
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

	# Create a learning rate scheduler with warmup followed by cosine annealing
	warmup_epochs = SCHEDULER_WARMUP_EPOCHS

	# Linear warmup scheduler - starts from 10% of base LR and increases linearly
	warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
		optimizer,
		start_factor=0.1,  # Start with 10% of the base learning rate
		end_factor=1.0,    # End with the full learning rate
		total_iters=warmup_epochs
	)

	# Cosine annealing scheduler for after warmup
	cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=EPOCHS - warmup_epochs,
		eta_min=1e-6
	)

	# Combine the schedulers using SequentialLR
	scheduler = torch.optim.lr_scheduler.SequentialLR(
		optimizer,
		schedulers=[warmup_scheduler, cosine_scheduler],
		milestones=[warmup_epochs]
	)

	# Define the MSE loss (without reduction) to optionally apply custom weighting.
	mse_loss = nn.MSELoss(reduction="none")

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
			# x_t = sqrt(alpha_bar)*action_seq + sqrt(1 - alpha_bar)*noise
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
				"batch_grad_norm": grad_norm,  # Log per-batch gradient norm
				"global_step": epoch * len(dataloader) + batch_idx
			})

		# Compute average loss for the epoch.
		avg_loss = running_loss / len(dataset)
		avg_grad_norm = running_grad_norm / batch_count if batch_count > 0 else 0.0

		# Log progress to console
		print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}, Grad Norm: {avg_grad_norm:.6f}")

		# Log metrics to WandB.
		current_lr = optimizer.param_groups[0]['lr']
		wandb.log({
			"epoch": epoch+1,
			"avg_loss": avg_loss,
			"avg_grad_norm": avg_grad_norm,  # Log epoch average gradient norm
			"learning_rate": current_lr
		})

		# Update the learning rate scheduler.
		scheduler.step()

		# Save a checkpoint every 10 epochs.
		if (epoch + 1) % 50 == 0:
			torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")

		# Validate the policy periodically.
		if (epoch + 1) % VALIDATION_INTERVAL == 0:
			val_reward, _, _ = validate_policy(model, device)  # Ignore the video path and action plot path
			print(f"Validation total reward at epoch {epoch+1}: {val_reward}")

	# Save the final model after training completes.
	torch.save(model.state_dict(), OUTPUT_DIR + "diffusion_policy.pth")
	print(f"Training complete. Model saved as {OUTPUT_DIR}diffusion_policy.pth.")

if __name__ == "__main__":
	train()
	wandb.finish()

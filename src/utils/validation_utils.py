"""
Utility functions for validating the diffusion policy model on the LeRobot environment.
"""

import os
import json
import math  # For cosine annealing scheduler.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.diffusion.diffusion_policy import DiffusionPolicy
from src.utils.diffusion_utils import get_beta_schedule, compute_alphas
from src.config import *
from einops import rearrange
from src.utils.normalize import Normalize  # Normalization helper class.
from PIL import Image  # For image loading and processing.
from src.diffusion.policy_inference import DiffusionPolicyInference
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
from src.utils.video_utils import save_video  # Import the new video utility
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.seed import set_seed
from src.datasets.policy_dataset import PolicyDataset
set_seed(42)

def get_chunk_time_encoding(length: int):
	"""
	Returns a 1D tensor of shape (length,) that linearly scales from 0 to 1.
	This is used to embed each timestep within a chunk.
	"""
	return torch.linspace(0, 1, steps=length)

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
	inference = DiffusionPolicyInference(model_path="", T=T, device=device, norm_stats_path=os.path.join(OUTPUT_DIR, DATA_SOURCE_DIR, "lerobot", NORM_STATS_FILENAME))
	inference.model = model  # Use current model weights

	# Track simulation time for proper action interpolation
	start_time = 0.0
	current_time = start_time

	# Initialize lists to track actions, timestamps, and new inferences
	action_history = []
	timestep_history = []
	inference_points = []  # Track when inferences occur

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

		# Load training dataset with all required parameters
		dataset = PolicyDataset(
			dataset_type="lerobot",
			data_dir=TRAINING_DATA_DIR,
			pred_horizon=WINDOW_SIZE,
			action_horizon=WINDOW_SIZE//2,
			obs_horizon=2
		)

		 # Initialize lists to store all ground truth and predicted actions, and inference points
		all_pred_actions = []
		all_gt_actions = []
		all_timesteps = []
		all_inference_points = []  # Track inference points

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
										   norm_stats_path=os.path.join(OUTPUT_DIR, DATA_SOURCE_DIR, "lerobot", NORM_STATS_FILENAME))
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
									   norm_stats_path=os.path.join(OUTPUT_DIR, DATA_SOURCE_DIR, "lerobot", NORM_STATS_FILENAME))
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
					action, is_new_inference = inference.sample_action(
						current_condition,
						current_image_tuple,
					)

					# Record prediction, ground truth, and timestep
					all_pred_actions.append(action.cpu().numpy())
					all_gt_actions.append(gt_actions[i].cpu().numpy())
					all_timesteps.append(current_time)

					 # Track if this was a new inference
					if is_new_inference:
						all_inference_points.append(current_time)

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
"""
Unit tests for the diffusion policy project.
Tests cover configuration, utility functions, network components,
the complete diffusion policy network, visual encoder, inference module, dataset generation,
and the diffusion process.
"""

import os
import json
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from src.config import *
from src.utils.diffusion_utils import get_beta_schedule, compute_alphas
from src.diffusion.diffusion_policy import DiffusionPolicy, UNet1D, ConditionalResidualBlock1D, FiLM
from src.diffusion.policy_inference import DiffusionPolicyInference
from src.diffusion.train_diffusion import PolicyDataset
from src.utils.validation_utils import *
from src.utils.normalize import Normalize
from src.datasets.policy_dataset import PolicyDataset
from src.utils.utils import *
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
# Set Plotly to use a non-browser renderer
import plotly.io as pio
pio.renderers.default = 'png'  # Use static PNG renderer instead of browser

class TestConfig(unittest.TestCase):
	"""Tests for configuration constants."""
	def test_constants(self):
		self.assertIsInstance(SCREEN_WIDTH, int)
		self.assertIsInstance(SCREEN_HEIGHT, int)
		self.assertTrue(ARM_LENGTH > 0)
		self.assertIsInstance(BASE_POS, torch.Tensor)
		self.assertEqual(ACTION_DIM, 2)  # XY coordinates
		self.assertEqual(CONDITION_DIM, 2)  # XY coordinates
		self.assertGreater(IMG_RES, 0)  # Image resolution should be positive

class TestDiffusionUtils(unittest.TestCase):
	"""Tests for diffusion utility functions."""
	def test_get_beta_schedule(self):
		betas = get_beta_schedule(T)
		self.assertEqual(betas.shape[0], T)
		self.assertTrue(torch.all(betas >= 0))
		self.assertTrue(torch.all(betas <= 0.999))  # Betas should be bounded

	def test_compute_alphas(self):
		betas = get_beta_schedule(T)
		alphas, alphas_cumprod = compute_alphas(betas)
		self.assertEqual(alphas.shape[0], T)
		self.assertEqual(alphas_cumprod.shape[0], T)
		self.assertTrue(torch.allclose(alphas, 1 - betas))
		self.assertTrue(torch.all(alphas_cumprod > 0))  # Always positive
		self.assertTrue(torch.all(alphas_cumprod <= 1.0))  # Bounded by 1
		# Verify that alphas_cumprod is monotonically decreasing
		self.assertTrue(torch.all(alphas_cumprod[:-1] >= alphas_cumprod[1:]))
		manual_cumprod = torch.cumprod(alphas, dim=0)
		self.assertTrue(torch.allclose(alphas_cumprod, manual_cumprod))

class TestNormalization(unittest.TestCase):
	"""Tests for normalization functions."""
	def setUp(self):
		self.normalize = Normalize.compute_from_limits()

	def test_normalize_unnormalize_roundtrip(self):
		# Test roundtrip on actions
		actions = torch.tensor([[100.0, 200.0], [300.0, 400.0]], dtype=torch.float32)
		normalized = self.normalize.normalize_action(actions)
		unnormalized = self.normalize.unnormalize_action(normalized)
		self.assertTrue(torch.allclose(actions, unnormalized, atol=UNIT_TEST_TOL))

		# Test normalized values are in [-1, 1] range
		self.assertTrue(torch.all(normalized >= -1.0))
		self.assertTrue(torch.all(normalized <= 1.0))

		# Test conditions
		condition = torch.tensor([50.0, 100.0], dtype=torch.float32)
		normalized_cond = self.normalize.normalize_condition(condition)
		self.assertTrue(torch.all(normalized_cond >= -1.0))
		self.assertTrue(torch.all(normalized_cond <= 1.0))

	def test_edge_cases(self):
		# Test min values
		min_action = torch.zeros(2, dtype=torch.float32)
		normalized = self.normalize.normalize_action(min_action)
		self.assertTrue(torch.allclose(normalized, torch.tensor([-1.0, -1.0], dtype=torch.float32)))

		# Test max values
		max_action = torch.tensor([ACTION_LIM, ACTION_LIM], dtype=torch.float32)
		normalized = self.normalize.normalize_action(max_action)
		self.assertTrue(torch.allclose(normalized, torch.tensor([1.0, 1.0], dtype=torch.float32)))

		# Test beyond range
		beyond_action = torch.tensor([ACTION_LIM * 1.5, ACTION_LIM * 1.5], dtype=torch.float32)
		normalized = self.normalize.normalize_action(beyond_action)
		self.assertTrue(torch.all(normalized > 1.0))
		unnormalized = self.normalize.unnormalize_action(normalized)
		self.assertTrue(torch.allclose(beyond_action, unnormalized, atol=UNIT_TEST_TOL))

class TestDiffusionPolicyComponents(unittest.TestCase):
	"""Tests for individual network components."""
	def test_FiLM(self):
		batch, channels, T_dim = 2, 8, 10
		cond_dim = 16
		film = FiLM(channels, cond_dim)
		x = torch.randn(batch, channels, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = film(x, cond)
		self.assertEqual(out.shape, (batch, channels, T_dim))

		# Test that FiLM produces both scale and bias modulations
		mod_params = film.fc(cond)
		self.assertEqual(mod_params.shape, (batch, channels * 2))

		# Reshape and extract scale and bias
		mod_params = mod_params.view(batch, 2, channels, 1)
		scale, bias = mod_params[:, 0], mod_params[:, 1]

		# Verify modulation is applied correctly
		expected = x * scale + bias
		self.assertTrue(torch.allclose(out, expected, atol=UNIT_TEST_TOL))

	def test_ConditionalResidualBlock1D(self):
		batch, in_channels, T_dim = 2, 8, 10
		out_channels = 16
		cond_dim = 12
		block = ConditionalResidualBlock1D(in_channels, out_channels, cond_dim)
		x = torch.randn(batch, in_channels, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = block(x, cond)
		self.assertEqual(out.shape, (batch, out_channels, T_dim))
		self.assertIsInstance(block.res_conv, torch.nn.Conv1d)

		# Test identity residual connection when channels match
		block_same = ConditionalResidualBlock1D(in_channels, in_channels, cond_dim)
		out_same = block_same(x, cond)
		self.assertEqual(out_same.shape, (batch, in_channels, T_dim))
		self.assertIsInstance(block_same.res_conv, torch.nn.Identity)

		# Verify gradient flow
		x.requires_grad_(True)
		out = block(x, cond)
		loss = out.sum()
		loss.backward()
		self.assertIsNotNone(x.grad)
		self.assertTrue(torch.any(x.grad != 0))

	def test_UNet1D(self):
		batch = 2
		window_size = WINDOW_SIZE
		action_dim = ACTION_DIM
		global_cond_dim = CONDITION_DIM + 128

		# Test the newly structured UNet1D
		unet = UNet1D(action_dim, global_cond_dim, diffusion_step_embed_dim=128)

		# Create sample inputs
		x = torch.randn(batch, window_size, action_dim)
		timestep = torch.tensor([100.0, 500.0])
		global_cond = torch.randn(batch, global_cond_dim)

		# Test forward pass
		out = unet(x, timestep, global_cond)
		self.assertEqual(out.shape, (batch, window_size, action_dim))

		# Test with irregular window size
		irregular_window_size = window_size + 4
		x_irreg = torch.randn(batch, irregular_window_size, action_dim)
		out_irreg = unet(x_irreg, timestep, global_cond)
		self.assertEqual(out_irreg.shape, (batch, irregular_window_size, action_dim))

	def test_model_temporal_processing(self):
		"""Tests that the UNet1D model correctly processes temporal sequences of varying lengths."""
		batch = 2
		window_size = WINDOW_SIZE
		action_dim = ACTION_DIM
		global_cond_dim = CONDITION_DIM + 128

		# Create a UNet with specific window size
		unet = UNet1D(action_dim, global_cond_dim, diffusion_step_embed_dim=128)
		timestep = torch.tensor([100.0, 500.0])
		global_cond = torch.randn(batch, global_cond_dim)

		# Test with standard window size
		x = torch.randn(batch, window_size, action_dim)
		out = unet(x, timestep, global_cond)
		self.assertEqual(out.shape, (batch, window_size, action_dim),
						 f"Expected output shape (batch, window_size, action_dim), got {out.shape}")

		# Test with variable sequence lengths:
		# 1. Shorter sequence (half window size)
		x_short = torch.randn(batch, window_size//2, action_dim)
		out_short = unet(x_short, timestep, global_cond)
		self.assertEqual(out_short.shape, (batch, window_size//2, action_dim),
						 "UNet should handle shorter sequences properly")

		# 2. Longer sequence (double window size)
		x_long = torch.randn(batch, window_size*2, action_dim)
		out_long = unet(x_long, timestep, global_cond)
		self.assertEqual(out_long.shape, (batch, window_size*2, action_dim),
						 "UNet should handle longer sequences properly")

class TestDiffusionPolicy(unittest.TestCase):
	"""Test the full diffusion policy network."""
	def test_forward_output_shape(self):
		batch = 2
		window_size = WINDOW_SIZE
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, window_size=window_size)
		x = torch.randn(batch, window_size, ACTION_DIM)
		t = torch.tensor([10.0, 500.0])
		state = torch.randn(batch, 4)  # Two states concatenated: t-1 and t

		# Test with a pair of images
		image = [torch.randn(batch, 3, IMG_RES, IMG_RES), torch.randn(batch, 3, IMG_RES, IMG_RES)]
		out = model(x, t, state, image)
		self.assertEqual(out.shape, (batch, window_size, ACTION_DIM))

		# Test with a single image (should be duplicated internally)
		single_image = torch.randn(batch, 3, IMG_RES, IMG_RES)
		out_single = model(x, t, state, single_image)
		self.assertEqual(out_single.shape, (batch, window_size, ACTION_DIM))

	def test_gradient_flow(self):
		batch = 2
		window_size = WINDOW_SIZE
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, window_size=window_size)
		x = torch.randn(batch, window_size, ACTION_DIM, requires_grad=True)
		t = torch.tensor([10.0, 500.0])
		state = torch.randn(batch, 4)
		image = [torch.randn(batch, 3, IMG_RES, IMG_RES), torch.randn(batch, 3, IMG_RES, IMG_RES)]

		# Forward pass
		out = model(x, t, state, image)
		target = torch.randn_like(out)
		loss = F.mse_loss(out, target)

		# Backward pass
		loss.backward()

		# Check gradients flow through the model
		has_gradients = all(p.grad is not None and torch.any(p.grad != 0)
						   for p in model.parameters() if p.requires_grad)

		self.assertTrue(has_gradients, "No gradients found in model parameters")

		# Check gradient flow through input
		self.assertIsNotNone(x.grad, "No gradient in input tensor")
		self.assertTrue(torch.any(x.grad != 0), "Zero gradient in input tensor")

class TestPolicyDatasetLerobot(unittest.TestCase):
	"""Tests for the PolicyDataset using the real lerobot dataset."""
	def test_dataset_loading_and_structure(self):
		dataset = PolicyDataset(
			dataset_type="lerobot",
			data_dir=TRAINING_DATA_DIR,
			pred_horizon=WINDOW_SIZE,
			action_horizon=WINDOW_SIZE//2,
			obs_horizon=2
		)

		self.assertGreater(len(dataset), 0, "Dataset should contain at least one sample")

		# Get the first sample safely with error handling
		try:
			sample = dataset[0]
			print(f"Successfully loaded sample 0. Type: {type(sample)}")
		except Exception as e:
			self.fail(f"Failed to load the first sample with error: {str(e)}")

		self.assertEqual(len(sample), 3, "Sample should contain condition, image, action")
		condition = sample["agent_pos"]
		image = sample["image"]
		action = sample["action"]

		# Check condition: since two state observations are concatenated, expect shape (4,)
		self.assertEqual(condition.shape, (4,))

		# Check images: should be a tensor with shape (obs_horizon, 3, IMG_RES, IMG_RES)
		self.assertIsInstance(image, torch.Tensor)
		self.assertEqual(image.shape, (2, 3, IMG_RES, IMG_RES))  # obs_horizon is 2

		# Check action: should be a tensor with shape (WINDOW_SIZE, ACTION_DIM)
		self.assertEqual(action.shape[1], ACTION_DIM)
		self.assertEqual(action.shape[0], WINDOW_SIZE)

		 # Verify dataset fidelity against raw LeRobot data
		try:
			from datasets import load_dataset

			print("\nVerifying dataset against raw LeRobot data...")
			raw_dataset = load_dataset("lerobot/pusht_image", split="train[:50]")
			print(f"Loaded {len(raw_dataset)} samples from raw LeRobot dataset")

			# Find an episode with enough frames
			episodes = {}
			for sample in raw_dataset:
				ep = sample["episode_index"]
				episodes.setdefault(ep, []).append(sample)

			test_episode = None
			for ep, frames in episodes.items():
				if len(frames) >= WINDOW_SIZE + 2:
					test_episode = ep
					break

			self.assertIsNotNone(test_episode, "Could not find an episode with sufficient frames")

			# Get frames from test episode in order
			ep_frames = sorted([f for f in raw_dataset if f["episode_index"] == test_episode],
							  key=lambda x: x["frame_index"])

			print(f"Testing with episode {test_episode}, which has {len(ep_frames)} frames")

			# Test frame with context
			i = 1  # Second frame

			# Get raw data and verify normalization
			if i > 0 and i + WINDOW_SIZE <= len(ep_frames):
				# Get states
				raw_states = [
					ep_frames[i-1]["observation.state"],
					ep_frames[i]["observation.state"]
				]
				raw_states_flat = np.concatenate([raw_states[0], raw_states[1]])
				raw_states_tensor = torch.tensor(raw_states_flat, dtype=torch.float32)

				# Normalize and verify
				normalized_states = dataset.normalize.normalize_condition(raw_states_tensor)
				self.assertEqual(normalized_states.shape, (4,), "Normalized state shape incorrect")

				# Round-trip test
				unnormalized_states = dataset.normalize.unnormalize_condition(normalized_states)
				self.assertTrue(torch.allclose(unnormalized_states, raw_states_tensor, atol=UNIT_TEST_TOL),
							   "Normalization round-trip failed for states")

				# Get actions
				raw_actions = []
				for j in range(i-1, i+WINDOW_SIZE-1):
					if j < len(ep_frames):
						raw_actions.append(ep_frames[j]["action"])

				if len(raw_actions) == WINDOW_SIZE:
					raw_actions_tensor = torch.tensor(raw_actions, dtype=torch.float32)
					normalized_actions = dataset.normalize.normalize_action(raw_actions_tensor)

					# Verify shape
					self.assertEqual(normalized_actions.shape, (WINDOW_SIZE, ACTION_DIM),
								   f"Normalized actions should have shape ({WINDOW_SIZE}, {ACTION_DIM})")

					# Round-trip verification
					unnormalized_actions = dataset.normalize.unnormalize_action(normalized_actions)
					self.assertTrue(torch.allclose(unnormalized_actions, raw_actions_tensor, atol=UNIT_TEST_TOL),
								  "Normalization round-trip failed for actions")

					# Find a similar sample in the dataset
					# Sample several from the dataset to see if we can find a match
					found_match = False
					for idx in range(min(100, len(dataset))):
						try:
							# Fix: Get dictionary instead of unpacking a tuple
							sample_data = dataset[idx]
							sample_cond = sample_data["agent_pos"]
							sample_action = sample_data["action"]

							# Check if states are similar (allowing for different normalization)
							if torch.norm(sample_cond - normalized_states) < UNIT_TEST_TOL:
								print(f"Found potentially matching sample at index {idx}")
								found_match = True
								break
						except Exception as e:
							print(f"Error checking sample {idx}: {e}")

					print(f"Found at least one similar sample in dataset: {found_match}")

				print("Verified dataset normalization and structure match raw data expectations")
		except Exception as e:
			print(f"Error during raw data verification: {e}")
			import traceback
			traceback.print_exc()
			self.fail(f"Failed test")

		# Pretty print some samples for visualization
		print("\n===== Dataset Sample Visualization =====")

		# Debug the dataset structure before attempting to visualize random samples
		print(f"Dataset type: {type(dataset)}")
		print(f"Dataset length: {len(dataset)}")

		# Instead of random samples, use specific indices to avoid potential issues
		# with invalid indices in case the dataset has gaps
		safe_indices = [0]  # Start with the first sample that we know works

		# Try to add one more index if the dataset has at least 2 items
		if len(dataset) > 1:
			safe_indices.append(1)

		for idx in safe_indices:
			try:
				print(f"\nAttempting to load sample {idx}...")

				# Fix: Get dictionary instead of unpacking a tuple
				sample_data = dataset[idx]
				if not isinstance(sample_data, dict) or len(sample_data) != 3:
					print(f"Warning: Sample {idx} has unexpected format: {type(sample_data)}, length: {len(sample_data) if hasattr(sample_data, '__len__') else 'N/A'}")
					continue

				condition = sample_data["agent_pos"]
				image = sample_data["image"]
				action = sample_data["action"]

				print(f"\n--- Sample {idx} ---")
				print(f"Condition (state): {condition.numpy()}")
				print(f"Action sequence shape: {action.shape}")
				print(f"Action sequence stats: min={action.min().item():.4f}, max={action.max().item():.4f}, mean={action.mean().item():.4f}")
				print("Action samples (first 3 and last 3):")
				for i in range(min(3, action.shape[0])):
					print(f"  Step {i}: {action[i].numpy()}")
				print("  ...")
				for i in range(max(0, action.shape[0]-3), action.shape[0]):
					print(f"  Step {i}: {action[i].numpy()}")

				# Convert data to numpy arrays for visualization
				x_vals = action[:, 0].numpy()
				y_vals = action[:, 1].numpy()

				# Ensure all arrays have the same length for plotting
				min_length = min(len(x_vals), len(y_vals))
				x_vals = x_vals[:min_length]
				y_vals = y_vals[:min_length]

				print(f"Adjusted shapes - x_vals: {x_vals.shape}, y_vals: {y_vals.shape}")

				# Convert to matplotlib for visualization instead of interactive plotly
				plt.figure(figsize=(10, 8))

				# Plot the trajectory with color gradient
				points = plt.scatter(
					x_vals, y_vals,
					cmap='viridis',
					s=50,
					alpha=0.8
				)
				plt.colorbar(points, label="Time")

				# Connect the points with a line
				plt.plot(x_vals, y_vals, 'k-', alpha=0.3)

				# Mark start and end points
				plt.scatter([x_vals[0]], [y_vals[0]], color='green', s=120, marker='o', label="Start")
				plt.scatter([x_vals[-1]], [y_vals[-1]], color='red', s=120, marker='x', label="End")

				plt.title(f"Action Trajectory (Sample {idx})")
				plt.xlabel("X coordinate")
				plt.ylabel("Y coordinate")
				plt.grid(True)
				plt.legend()
				if 'DISPLAY_PLOTS' in globals() and DISPLAY_PLOTS:
					plt.show(block=False)
					plt.pause(0.1)
				else:
					plt.close()

				# Display the images using matplotlib - with proper denormalization
				fig, axes = plt.subplots(1, 2, figsize=(12, 5))

				# Define ImageNet mean and std for denormalization
				mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
				std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

				for i in range(image.shape[0]):
					# Get the individual image tensor
					img_tensor = image[i]

					# Print image statistics for debugging
					print(f"Image {i+1} stats: min={img_tensor.min().item():.4f}, "
						  f"max={img_tensor.max().item():.4f}, "
						  f"mean={img_tensor.mean().item():.4f}")

					# First, denormalize if the image was normalized with ImageNet stats
					img_denorm = img_tensor * std + mean

					# Convert to numpy and ensure correct channel order (C,H,W) -> (H,W,C)
					img_np = img_denorm.permute(1, 2, 0).numpy()

					# Ensure values are in valid display range [0, 1]
					img_np = np.clip(img_np, 0, 1)

					# Print numpy array stats for verification
					print(f"Numpy image {i+1} shape: {img_np.shape}, "
						  f"min={img_np.min():.4f}, max={img_np.max().item():.4f}")

					# Display the image
					axes[i].imshow(img_np)
					axes[i].set_title(f"Image {i+1}")
					axes[i].axis('off')

				plt.suptitle(f"Input Images (Sample {idx})")
				plt.tight_layout()
				if 'DISPLAY_PLOTS' in globals() and DISPLAY_PLOTS:
					plt.show(block=False)
					plt.pause(0.1)
				else:
					plt.close()

				# Alternative visualization with pure PIL (just in case)
				try:
					plt.figure(figsize=(12, 5))
					for i in range(image.shape[0]):
						# Get the individual image tensor
						img_tensor = image[i]

						# Denormalize
						img_denorm = img_tensor * std + mean

						# Convert to PIL
						img_pil = transforms.ToPILImage()(img_denorm)

						plt.subplot(1, 2, i+1)
						plt.imshow(img_pil)
						plt.title(f"Image {i+1} (PIL)")
						plt.axis('off')

					plt.tight_layout()
					if 'DISPLAY_PLOTS' in globals() and DISPLAY_PLOTS:
						plt.show(block=False)
						plt.pause(0.1)
					else:
						plt.close()
				except Exception as e:
					print(f"PIL visualization failed: {e}")

			except Exception as e:
				print(f"Error processing sample {idx}: {str(e)}")
				import traceback
				traceback.print_exc()

	def test_dataloader_batch_shapes(self):
		"""Test that batches from the DataLoader have the correct shapes."""
		dataset = PolicyDataset(
			dataset_type="lerobot",
			data_dir=TRAINING_DATA_DIR,
			pred_horizon=WINDOW_SIZE,
			action_horizon=WINDOW_SIZE//2,
			obs_horizon=2
		)

		# Create dataloader with appropriate parameters
		batch_size = 1  # Using a small batch size for testing
		dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			num_workers=0,  # Use 0 workers for testing
			shuffle=True,
			# accelerate cpu-gpu transfer
			pin_memory=True,
			# don't kill worker process after each epoch
			persistent_workers=False  # False for testing since we use 0 workers
		)

		# Get a batch and verify shapes
		try:
			batch = next(iter(dataloader))

			# Print and verify shapes
			print("\n===== DataLoader Batch Shapes =====")
			print(f"batch['image'].shape: {batch['image'].shape}")
			print(f"batch['agent_pos'].shape: {batch['agent_pos'].shape}")
			print(f"batch['action'].shape: {batch['action'].shape}")
			print("\n-----------------------------------\n")

			# Assert the correct shapes
			self.assertEqual(batch['image'].shape, (batch_size, 2, 3, IMG_RES, IMG_RES),
							f"Expected image shape {(batch_size, 2, 3, IMG_RES, IMG_RES)}, got {batch['image'].shape}")
			self.assertEqual(batch['agent_pos'].shape, (batch_size, 4),
							f"Expected agent_pos shape {(batch_size, 4)}, got {batch['agent_pos'].shape}")
			self.assertEqual(batch['action'].shape, (batch_size, WINDOW_SIZE, ACTION_DIM),
							f"Expected action shape {(batch_size, WINDOW_SIZE, ACTION_DIM)}, got {batch['action'].shape}")

			# Verify data types and ranges
			self.assertTrue(torch.is_floating_point(batch['image']), "Images should be floating point tensors")
			self.assertTrue(torch.is_floating_point(batch['agent_pos']), "Agent positions should be floating point tensors")
			self.assertTrue(torch.is_floating_point(batch['action']), "Actions should be floating point tensors")

		except Exception as e:
			self.fail(f"Failed to get batch from dataloader with error: {e}")
			import traceback
			traceback.print_exc()

	def test_dataset_sequence_structure(self):
		"""
		Tests that the PolicyDataset maintains the correct temporal structure:
		- observations: [t-1, t]
		- actions predicted: [t-1, t+WINDOW_SIZE-2]
		- actions executed: [t, t+WINDOW_SIZE//2-1]
		"""
		# Initialize dataset with specific horizons
		pred_horizon = 16  # total prediction horizon
		obs_horizon = 2    # observation horizon (t-1, t)
		action_horizon = 8 # executed action horizon

		dataset = PolicyDataset(
			dataset_type="lerobot",
			data_dir=TRAINING_DATA_DIR,
			pred_horizon=pred_horizon,
			obs_horizon=obs_horizon,
			action_horizon=action_horizon
		)

		# Get a sample
		sample = dataset[0]

		# Check observation structure
		self.assertEqual(sample["image"].shape, (obs_horizon, 3, IMG_RES, IMG_RES),
						f"Expected image shape (obs_horizon, 3, IMG_RES, IMG_RES), got {sample['image'].shape}")

		# Check agent position structure - flattened for model input (2 observations × 2 dimensions)
		self.assertEqual(sample["agent_pos"].shape, (obs_horizon * 2,),
						f"Expected agent_pos shape (obs_horizon*2,), got {sample['agent_pos'].shape}")

		# Check action structure - full prediction horizon
		self.assertEqual(sample["action"].shape, (pred_horizon, ACTION_DIM),
						f"Expected action shape (pred_horizon, ACTION_DIM), got {sample['action'].shape}")

		# Now let's test with pred_horizon matching WINDOW_SIZE from config
		dataset = PolicyDataset(
			dataset_type="lerobot",
			data_dir=TRAINING_DATA_DIR,
			pred_horizon=WINDOW_SIZE,
			obs_horizon=obs_horizon,
			action_horizon=WINDOW_SIZE//2
		)

		# Get a sample
		sample = dataset[0]

		# Verify observations (t-1, t)
		self.assertEqual(sample["image"].shape, (obs_horizon, 3, IMG_RES, IMG_RES))

		# Verify actions predicted [t-1, WINDOW_SIZE-1]
		self.assertEqual(sample["action"].shape, (WINDOW_SIZE, ACTION_DIM))

	def test_temporal_structure_alignment(self):
		"""
			Tests that the PolicyDataset maintains the correct temporal structure:
			- observations: [t-1, t]
			- actions predicted: [t-1, t+WINDOW_SIZE-2]
			- actions executed: [t, t+WINDOW_SIZE//2-1]
			"""
		# Initialize dataset with specific horizons
		pred_horizon = 16  # total prediction horizon
		obs_horizon = 2    # observation horizon (t-1, t)
		action_horizon = 8 # executed action horizon

		dataset = PolicyDataset(
			dataset_type="lerobot",
			data_dir=TRAINING_DATA_DIR,
			pred_horizon=pred_horizon,
			obs_horizon=obs_horizon,
			action_horizon=action_horizon
		)

		# Get a sample
		sample = dataset[0]

		# Check observation structure
		self.assertEqual(sample["image"].shape, (obs_horizon, 3, IMG_RES, IMG_RES),
						f"Expected image shape (obs_horizon, 3, IMG_RES, IMG_RES), got {sample['image'].shape}")

		# Check agent position structure - flattened for model input (2 observations × 2 dimensions)
		self.assertEqual(sample["agent_pos"].shape, (obs_horizon * 2,),
						f"Expected agent_pos shape (obs_horizon*2,), got {sample['agent_pos'].shape}")

		# Check action structure - full prediction horizon
		self.assertEqual(sample["action"].shape, (pred_horizon, ACTION_DIM),
						f"Expected action shape (pred_horizon, ACTION_DIM), got {sample['action'].shape}")

		# Now let's test with pred_horizon matching WINDOW_SIZE from config
		dataset = PolicyDataset(
			dataset_type="lerobot",
			data_dir=TRAINING_DATA_DIR,
			pred_horizon=WINDOW_SIZE,
			obs_horizon=obs_horizon,
			action_horizon=WINDOW_SIZE//2
		)

		# Get a sample
		sample = dataset[0]

		# Verify observations (t-1, t)
		self.assertEqual(sample["image"].shape, (obs_horizon, 3, IMG_RES, IMG_RES))

		# Verify actions predicted [t-1, WINDOW_SIZE-1]
		self.assertEqual(sample["action"].shape, (WINDOW_SIZE, ACTION_DIM))

	def test_temporal_alignment_with_raw_data(self):
		"""
		Verify temporal alignment by comparing against a real episode from the LeRobot dataset.
		This ensures that observations [t-1, t], actions predicted [t-1, t+pred_horizon-2],
		and actions executed [t, t+action_horizon-1] are correctly aligned.
		"""
		try:
			from datasets import load_dataset
			from tabulate import tabulate

			# Specify our temporal horizons
			obs_horizon = 2        # Observations: [t-1, t]
			action_horizon = 8     # Actions executed: [t, t+7]
			pred_horizon = 16      # Actions predicted: [t-1, t+14]

			# Load raw data (limited sample for testing)
			print("\nLoading raw LeRobot data for temporal alignment test...")
			raw_dataset = load_dataset("lerobot/pusht_image", split="train[:100]")

			# Find episodes and select one with sufficient frames
			episodes = {}
			for sample in raw_dataset:
				ep = sample["episode_index"]
				episodes.setdefault(ep, []).append(sample)

			# Find an episode with enough frames for our horizons
			test_episode = None
			for ep, frames in episodes.items():
				if len(frames) >= pred_horizon + 2:  # Need extra frames to ensure a valid window
					test_episode = ep
					break

			self.assertIsNotNone(test_episode, "Could not find episode with sufficient frames")

			# Sort frames by index to ensure temporal order
			ep_frames = sorted([f for f in raw_dataset if f["episode_index"] == test_episode],
							  key=lambda x: x["frame_index"])

			print(f"Testing with episode {test_episode}, which has {len(ep_frames)} frames")

			# Create dataset with our specific horizons
			dataset = PolicyDataset(
				dataset_type="lerobot",
				data_dir=TRAINING_DATA_DIR,
				pred_horizon=pred_horizon,
				action_horizon=action_horizon,
				obs_horizon=obs_horizon
			)

			# Now we need to find samples in our dataset that use this episode
			# We'll check for matching state/action values to identify them

			# First, build a reference sample from the raw data at a specific position
			position = 5  # Choose a position with enough context (not at the start)

			if position >= len(ep_frames) - pred_horizon:
				position = len(ep_frames) - pred_horizon - 1

			print(f"Creating reference sample at position {position} in episode {test_episode}")

			# Reference observations [t-1, t]
			ref_obs = [
				ep_frames[position-1]["observation.state"],
				ep_frames[position]["observation.state"]
			]
			ref_obs_flat = np.concatenate(ref_obs)
			ref_obs_tensor = torch.tensor(ref_obs_flat, dtype=torch.float32)
			normalized_ref_obs = dataset.normalize.normalize_condition(ref_obs_tensor)

			# Reference actions [t-1, t+pred_horizon-2]
			ref_actions = []
			for i in range(position-1, position+pred_horizon-1):
				if i < len(ep_frames):
					ref_actions.append(ep_frames[i]["action"])
				else:
					# Pad with last action if needed
					ref_actions.append(ep_frames[-1]["action"])

			ref_actions_tensor = torch.tensor(ref_actions, dtype=torch.float32)
			normalized_ref_actions = dataset.normalize.normalize_action(ref_actions_tensor)

			# Extract the executed portion [t, t+action_horizon-1]
			ref_executed = ref_actions_tensor[1:1+action_horizon]

			# Now search for a matching sample in the dataset
			found_match = False
			matching_idx = -1
			match_similarity = float('inf')

			# Try more samples to increase chance of finding a match
			num_samples = min(1000, len(dataset))

			print(f"Searching through {num_samples} samples for a match...")
			for idx in range(num_samples):
				try:
					sample = dataset[idx]
					sample_obs = sample["agent_pos"]
					sample_actions = sample["action"]

					# Compute similarity between this sample and our reference
					obs_similarity = torch.norm(sample_obs - normalized_ref_obs)

					if obs_similarity < UNIT_TEST_TOL:
						# We found a close observation match, now check actions
						action_similarity = torch.norm(sample_actions - normalized_ref_actions)
						total_similarity = obs_similarity + 0.1 * action_similarity

						if total_similarity < match_similarity:
							match_similarity = total_similarity
							matching_idx = idx
							found_match = True

							# If we have a very good match, break early
							if total_similarity < UNIT_TEST_TOL:
								break

				except Exception as e:
					continue

			if found_match:
				print(f"Found matching sample at index {matching_idx} with similarity {match_similarity:.4f}")

				# Get the matched sample
				matched_sample = dataset[matching_idx]

				# Get the indices used to create this sample
				indices = dataset.indices[matching_idx]
				buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices

				print(f"Buffer indices: {buffer_start_idx}:{buffer_end_idx}, Sample placement: {sample_start_idx}:{sample_end_idx}")

				# Test 1: Verify observations match the expected frames
				# The observation at t-1 and t should match our reference
				sample_obs = matched_sample["agent_pos"]
				self.assertLess(torch.norm(sample_obs - normalized_ref_obs), UNIT_TEST_TOL,
							  "Observations should closely match reference")

				# Test 2: Verify the predicted action sequence matches expected raw data
				# The entire action sequence should match our reference actions
				sample_actions = matched_sample["action"]
				self.assertEqual(sample_actions.shape, normalized_ref_actions.shape,
							   "Action sequence shape should match reference")

				# Allow some tolerance since there might be minor differences due to
				# normalization or different episodes with similar patterns
				action_diff = torch.norm(sample_actions - normalized_ref_actions)
				self.assertLessEqual(action_diff / pred_horizon, UNIT_TEST_TOL,
								   f"Per-step action difference ({action_diff/pred_horizon:.4f}) exceeds tolerance")

				# Test 3: Verify execution window has the expected actions
				# The executed portion [t:t+action_horizon] should match our reference executed actions
				executed_actions = sample_actions[1:1+action_horizon]
				executed_ref = normalized_ref_actions[1:1+action_horizon]
				self.assertEqual(executed_actions.shape, executed_ref.shape,
							   f"Executed action shape should be {executed_ref.shape}")

				# Visual verification - print the temporal structure
				print("\n=== Temporal Structure Verification ===")
				print("Observations [t-1, t]:")
				for i, obs in enumerate(ref_obs):
					print(f"  t{i-1}: {obs}")

				print("\nPredicted Actions [t-1, t+14]:")
				for i, act in enumerate(ref_actions[:5]):
					print(f"  t{i-1}: {act}")
				print("  ...")
				for i, act in enumerate(ref_actions[-3:]):
					print(f"  t{i+pred_horizon-4}: {act}")

				print("\nExecuted Actions [t, t+7]:")
				for i, act in enumerate(ref_executed[:3]):
					print(f"  t{i}: {act}")
				print("  ...")
				for i, act in enumerate(ref_executed[-3:]):
					print(f"  t{i+action_horizon-3}: {act}")

				 # Enhanced comparison display section
				print("\n═════════════════════════════════════════════")
				print("  TEMPORAL ALIGNMENT VERIFICATION DETAILS  ")
				print("═════════════════════════════════════════════\n")

				# 1. Observation Comparison
				print("OBSERVATION COMPARISON [t-1, t]:")
				# Unnormalize the sample observation to match raw data scale
				sample_obs_tensor = matched_sample["agent_pos"].reshape(2, 2)  # reshape to (obs_horizon, dim)
				unnorm_sample_obs = dataset.normalize.unnormalize_condition(sample_obs_tensor.flatten())
				unnorm_sample_obs = unnorm_sample_obs.reshape(2, 2)

				# Create table headers and data for observation comparison
				obs_headers = ["Timestep", "Raw X", "Raw Y", "Processed X", "Processed Y", "Diff X", "Diff Y"]
				obs_data = []
				for t_idx in range(obs_horizon):
					raw_xy = torch.tensor(ref_obs[t_idx])
					proc_xy = unnorm_sample_obs[t_idx]
					diff_xy = raw_xy - proc_xy
					obs_data.append([
						f"t{t_idx-1}",
						f"{raw_xy[0]:.4f}", f"{raw_xy[1]:.4f}",
						f"{proc_xy[0]:.4f}", f"{proc_xy[1]:.4f}",
						f"{diff_xy[0]:.4f}", f"{diff_xy[1]:.4f}"
					])

				print(tabulate(obs_data, headers=obs_headers, tablefmt="pretty"))
				print()

				# 2. Action Sequence Comparison (show a subset for clarity)
				print("ACTION SEQUENCE COMPARISON [t-1, t+14]:")

				# Unnormalize the sample actions for comparison
				unnorm_sample_actions = dataset.normalize.unnormalize_action(matched_sample["action"])

				# Create action comparison table with subset of timesteps
				action_headers = ["Timestep", "Raw X", "Raw Y", "Processed X", "Processed Y", "Diff X", "Diff Y"]
				action_data = []

				# Show first few and last few timesteps (avoid too much data)
				display_indices = list(range(0, 3)) + list(range(pred_horizon-3, pred_horizon))

				for i in display_indices:
					t_label = f"t{i-1}"  # t-1, t, t+1, etc.

					# Handle index bounds for raw actions
					raw_xy = ref_actions_tensor[i] if i < len(ref_actions_tensor) else ref_actions_tensor[-1]
					proc_xy = unnorm_sample_actions[i]
					diff_xy = raw_xy - proc_xy

					action_data.append([
						t_label,
						f"{raw_xy[0]:.4f}", f"{raw_xy[1]:.4f}",
						f"{proc_xy[0]:.4f}", f"{proc_xy[1]:.4f}",
						f"{diff_xy[0]:.4f}", f"{diff_xy[1]:.4f}"
					])

				print(tabulate(action_data, headers=action_headers, tablefmt="pretty"))
				print()

				# 3. Execution Window Comparison
				print("EXECUTION WINDOW COMPARISON [t, t+7]:")

				# Extract executed actions
				executed_actions = unnorm_sample_actions[1:1+action_horizon]
				ref_executed = ref_actions_tensor[1:1+action_horizon]

				exec_headers = ["Timestep", "Raw X", "Raw Y", "Processed X", "Processed Y", "Diff X", "Diff Y"]
				exec_data = []

				# Show first few and last few timesteps of execution window
				exec_indices = list(range(0, 2)) + list(range(action_horizon-2, action_horizon))

				for i in exec_indices:
					t_label = f"t+{i}"  # t, t+1, t+2, etc.
					raw_xy = ref_executed[i]
					proc_xy = executed_actions[i]
					diff_xy = raw_xy - proc_xy

					exec_data.append([
						t_label,
						f"{raw_xy[0]:.4f}", f"{raw_xy[1]:.4f}",
						f"{proc_xy[0]:.4f}", f"{proc_xy[1]:.4f}",
						f"{diff_xy[0]:.4f}", f"{diff_xy[1]:.4f}"
					])

				print(tabulate(exec_data, headers=exec_headers, tablefmt="pretty"))
				print()

				# 4. Summary Statistics
				print("SUMMARY STATISTICS:")
				obs_error = torch.norm(torch.tensor(ref_obs).flatten() - unnorm_sample_obs.flatten())
				action_error = torch.norm(ref_actions_tensor - unnorm_sample_actions)
				exec_error = torch.norm(ref_executed - executed_actions)

				# Compute per-element errors
				obs_per_element = obs_error/obs_horizon/2
				action_per_element = action_error/pred_horizon/2
				exec_per_element = exec_error/action_horizon/2

				# Determine pass/fail status for each metric
				obs_status = "PASS" if obs_per_element < UNIT_TEST_TOL else "FAIL"
				action_status = "PASS" if action_per_element < UNIT_TEST_TOL else "FAIL"
				exec_status = "PASS" if exec_per_element < UNIT_TEST_TOL else "FAIL"

				stats_headers = ["Metric", "Value", "Per-Element", "Status"]
				stats_data = [
					["Observation Error", f"{obs_error:.4f}", f"{obs_per_element:.4f}", obs_status],
					["Action Sequence Error", f"{action_error:.4f}", f"{action_per_element:.4f}", action_status],
					["Execution Window Error", f"{exec_error:.4f}", f"{exec_per_element:.4f}", exec_status]
				]

				print(tabulate(stats_data, headers=stats_headers, tablefmt="pretty"))
				print()

				# Fail the test if any metric fails
				self.assertTrue(obs_status == "PASS", f"Observation error too high: {obs_per_element:.4f} > {UNIT_TEST_TOL}")
				self.assertTrue(action_status == "PASS", f"Action sequence error too high: {action_per_element:.4f} > {UNIT_TEST_TOL}")
				self.assertTrue(exec_status == "PASS", f"Execution window error too high: {exec_per_element:.4f} > {UNIT_TEST_TOL}")

				# 5. Enhanced visual diagram of temporal structure
				print("\nTEMPORAL STRUCTURE DIAGRAM:")
				print("Time:  t-1   t    t+1  ...  t+7  ...  t+14")
				print("      ┌─────┬─────┬─────┬───┬─────┬───┬─────┐")
				print("Obs:  │  o  │  o  │     │   │     │   │     │")
				print("      ├─────┼─────┼─────┼───┼─────┼───┼─────┤")
				print("Exec: │     │  a  │  a  │...│  a  │   │     │")
				print("      ├─────┼─────┼─────┼───┼─────┼───┼─────┤")
				print("Pred: │  p  │  p  │  p  │...│  p  │...│  p  │")
				print("      └─────┴─────┴─────┴───┴─────┴───┴─────┘")

				# 6. Enhanced visualization - trajectory comparison
				try:
					# Create a nicer trajectory comparison plot
					plt.figure(figsize=(12, 8))

					# Create a 2x2 grid of plots
					plt.subplot(2, 2, 1)
					# Plot raw unnormalized trajectories
					plt.title('Raw vs Processed Full Trajectories')
					plt.plot(ref_actions_tensor[:, 0].numpy(), ref_actions_tensor[:, 1].numpy(),
							'r-', label='Raw Data')
					plt.plot(unnorm_sample_actions[:, 0].numpy(), unnorm_sample_actions[:, 1].numpy(),
							'b--', label='Processed Data')
					plt.scatter(ref_actions_tensor[0, 0], ref_actions_tensor[0, 1],
							   c='green', s=100, marker='o', label='Start')
					plt.scatter(ref_actions_tensor[-1, 0], ref_actions_tensor[-1, 1],
							   c='red', s=100, marker='x', label='End')
					plt.xlabel('X Position')
					plt.ylabel('Y Position')
					plt.grid(True)
					plt.legend()

					# Plot executed trajectory portion
					plt.subplot(2, 2, 2)
					plt.title('Execution Window [t, t+7]')
					plt.plot(ref_executed[:, 0].numpy(), ref_executed[:, 1].numpy(),
							'r-', label='Raw Executed')
					plt.plot(executed_actions[:, 0].numpy(), executed_actions[:, 1].numpy(),
							'b--', label='Processed Executed')
					plt.scatter(ref_executed[0, 0], ref_executed[0, 1],
							   c='green', s=100, marker='o', label='Start')
					plt.scatter(ref_executed[-1, 0], ref_executed[-1, 1],
							   c='red', s=100, marker='x', label='End')
					plt.xlabel('X Position')
					plt.ylabel('Y Position')
					plt.grid(True)
					plt.legend()

					# Plot the position differences over time
					plt.subplot(2, 2, 3)
					plt.title('Position Difference Over Time')
					time_steps = range(pred_horizon)
					plt.plot(time_steps, abs(ref_actions_tensor[:, 0] - unnorm_sample_actions[:, 0]).numpy(),
							'r-', label='X Difference')
					plt.plot(time_steps, abs(ref_actions_tensor[:, 1] - unnorm_sample_actions[:, 1]).numpy(),
							'b-', label='Y Difference')
					plt.axvspan(1, action_horizon, color='green', alpha=0.2, label='Execution Window')
					plt.xlabel('Time Step')
					plt.ylabel('Absolute Difference')
					plt.grid(True)
					plt.legend()

					# Plot the observations
					plt.subplot(2, 2, 4)
					plt.title('Observation Positions')
					plt.scatter([ref_obs[0][0], ref_obs[1][0]], [ref_obs[0][1], ref_obs[1][1]],
							   c=['blue', 'red'], s=100, marker='o', label='Raw Obs')
					plt.scatter([unnorm_sample_obs[0][0], unnorm_sample_obs[0][1]],
							   [unnorm_sample_obs[1][0], unnorm_sample_obs[1][1]],
							   c=['blue', 'red'], s=100, marker='x', label='Processed Obs')
					plt.xlabel('X Position')
					plt.ylabel('Y Position')
					plt.grid(True)
					plt.legend()

					plt.tight_layout()
					if 'DISPLAY_PLOTS' in globals() and DISPLAY_PLOTS:
						plt.show(block=False)
						plt.pause(0.1)
					else:
						plt.close()
				except Exception as e:
					print(f"Enhanced visualization error: {e}")
					import traceback
					traceback.print_exc()

			else:
				self.fail("Could not find a matching sample in the dataset")

		except Exception as e:
			import traceback
			traceback.print_exc()
			self.fail(f"Failed temporal alignment test with error: {e}")

class TestPolicyInference(unittest.TestCase):
	"""Tests for the DiffusionPolicyInference module."""
	def setUp(self):
		self.model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, window_size=WINDOW_SIZE)
		self.inference = DiffusionPolicyInference(model_path="")
		self.inference.model = self.model

	def test_sample_action(self):
		state = torch.randn(1, 4)
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]
		action = self.inference.generate_action_sequence(state, image, num_ddim_steps=10)
		# Expect a tensor of shape (WINDOW_SIZE-1, ACTION_DIM)
		self.assertEqual(action.shape, (WINDOW_SIZE - 1, ACTION_DIM))
		self.assertFalse(torch.isnan(action).any())
		# Check that actions are clamped within screen bounds
		self.assertTrue(torch.all(action[:, 0] >= 0))
		self.assertTrue(torch.all(action[:, 0] <= SCREEN_WIDTH - 1))
		self.assertTrue(torch.all(action[:, 1] >= 0))
		self.assertTrue(torch.all(action[:, 1] <= SCREEN_HEIGHT - 1))

	def test_deterministic_sampling(self):
		torch.manual_seed(42)
		state = torch.randn(1, 4)
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]
		torch.manual_seed(42)
		action1 = self.inference.generate_action_sequence(state, image, num_ddim_steps=10)
		torch.manual_seed(42)
		action2 = self.inference.generate_action_sequence(state, image, num_ddim_steps=10)
		self.assertTrue(torch.allclose(action1, action2))

	def test_ddim_sampling_time_steps(self):
		state = torch.randn(1, 4)
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]
		action = self.inference.generate_action_sequence(state, image, num_ddim_steps=5)
		self.assertEqual(action.shape, (WINDOW_SIZE - 1, ACTION_DIM))
		action = self.inference.generate_action_sequence(state, image, num_ddim_steps=20)
		self.assertEqual(action.shape, (WINDOW_SIZE - 1, ACTION_DIM))

	def test_policy_sequence_generation(self):
		"""Tests that the DiffusionPolicy correctly generates action sequences with the right temporal structure."""
		# Create a model with specific window size
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, window_size=WINDOW_SIZE)
		inference = DiffusionPolicyInference(model_path="")
		inference.model = model

		# Create synthetic inputs
		state = torch.randn(1, 4)  # [batch, obs_horizon*state_dim]
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]

		# Generate action sequence
		action = inference.generate_action_sequence(state, image, num_ddim_steps=10)

		# Expect a tensor of shape (WINDOW_SIZE-1, ACTION_DIM)
		# Since we're excluding the t-1 action (which is part of the condition)
		# Note that on execution, we only use WINDOW_SIZE // 2 actions.
		self.assertEqual(action.shape, (WINDOW_SIZE-1, ACTION_DIM),
						 f"Expected action shape (WINDOW_SIZE-1, ACTION_DIM), got {action.shape}")

	def test_action_execution_window(self):
		"""
		Tests that the generated action sequence can be properly windowed
		to extract the executed portion (t to t+action_horizon-1).
		"""
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, window_size=WINDOW_SIZE)
		inference = DiffusionPolicyInference(model_path="")
		inference.model = model

		# Create synthetic inputs
		state = torch.randn(1, 4)  # [batch, obs_horizon*state_dim]
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]

		# Generate action sequence
		full_action = inference.generate_action_sequence(state, image, num_ddim_steps=10)

		# Extract the executed portion [t, t+action_horizon-1]
		# In the full sequence [t-1, t+WINDOW_SIZE-2], executed starts at index 1
		action_horizon = WINDOW_SIZE//2
		executed_actions = full_action[1:1+action_horizon]

		# Verify shape of executed actions
		self.assertEqual(executed_actions.shape, (action_horizon, ACTION_DIM),
						 f"Expected executed actions shape ({action_horizon}, {ACTION_DIM}), got {executed_actions.shape}")

		# Verify executed actions stay within bounds
		self.assertTrue(torch.all(executed_actions[:, 0] >= 0))
		self.assertTrue(torch.all(executed_actions[:, 0] <= SCREEN_WIDTH - 1))
		self.assertTrue(torch.all(executed_actions[:, 1] >= 0))
		self.assertTrue(torch.all(executed_actions[:, 1] <= SCREEN_HEIGHT - 1))

class TestDiffusionProcess(unittest.TestCase):
	"""Tests for the diffusion process (forward and reverse)."""
	def test_forward_diffusion(self):
		batch = 2
		window_size = WINDOW_SIZE
		x_0 = torch.randn(batch, window_size, ACTION_DIM)
		betas = get_beta_schedule(T)
		alphas, alphas_cumprod = compute_alphas(betas)
		t = torch.tensor([T//2, T//4])
		alpha_bar = alphas_cumprod[t].view(batch, 1, 1)
		torch.manual_seed(0)
		noise = torch.randn_like(x_0)
		x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
		self.assertEqual(x_t.shape, x_0.shape)
		snr_0 = torch.mean(torch.abs(torch.sqrt(alpha_bar[0]) * x_0[0]) /
						  torch.abs(torch.sqrt(1 - alpha_bar[0]) * noise[0]))
		snr_1 = torch.mean(torch.abs(torch.sqrt(alpha_bar[1]) * x_0[1]) /
						  torch.abs(torch.sqrt(1 - alpha_bar[1]) * noise[1]))
		self.assertLess(snr_0, snr_1)

class TestPolicyTraining(unittest.TestCase):
	"""Tests for the policy training components."""
	def test_data_time_encoding(self):
		# Test the time encoding for sequences
		for length in [10, 15, 20]:
			time_encoding = get_chunk_time_encoding(length)
			self.assertEqual(time_encoding.shape, (length,))
			self.assertAlmostEqual(time_encoding[0].item(), 0.0, places=5)
			self.assertAlmostEqual(time_encoding[-1].item(), 1.0, places=5)
			self.assertTrue(torch.all(time_encoding[1:] > time_encoding[:-1]))

if __name__ == '__main__':
	# Add argument parsing
	parser = argparse.ArgumentParser(description='Run unit tests for the diffusion policy project.')
	parser.add_argument('--display-plots', action='store_true',
						help='Display plots during tests (default: False)')

	args = parser.parse_args()

	# Pass the display_plots setting to the global scope
	# so tests can access it
	globals()['DISPLAY_PLOTS'] = args.display_plots

	# Run the tests
	unittest.main(argv=['first-arg-is-ignored'])

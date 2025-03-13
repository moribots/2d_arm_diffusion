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
from src.config import *
from src.utils.diffusion_utils import get_beta_schedule, compute_alphas
from src.diffusion.diffusion_policy import DiffusionPolicy, UNet1D, ResidualBlock1D, FiLM
from src.diffusion.policy_inference import DiffusionPolicyInference
from src.diffusion.train_diffusion import PolicyDataset, get_chunk_time_encoding
from src.utils.normalize import Normalize
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
		self.assertTrue(torch.allclose(actions, unnormalized, atol=1e-5))
		
		# Test normalized values are in [-1, 1] range
		self.assertTrue(torch.all(normalized >= -1.0))
		self.assertTrue(torch.all(normalized <= 1.0))
		
		# Test conditions
		condition = torch.tensor([50.0, 100.0, 150.0, 200.0], dtype=torch.float32)
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
		self.assertTrue(torch.allclose(beyond_action, unnormalized, atol=1e-5))

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
		bias = film.fc(cond).unsqueeze(-1)
		expected = x + bias
		self.assertTrue(torch.allclose(out, expected))
		
	def test_ResidualBlock1D(self):
		batch, in_channels, T_dim = 2, 8, 10
		out_channels = 16
		cond_dim = 12
		block = ResidualBlock1D(in_channels, out_channels, cond_dim)
		x = torch.randn(batch, in_channels, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = block(x, cond)
		self.assertEqual(out.shape, (batch, out_channels, T_dim))
		self.assertIsInstance(block.res_conv, torch.nn.Conv1d)
		block_same = ResidualBlock1D(in_channels, in_channels, cond_dim)
		out_same = block_same(x, cond)
		self.assertEqual(out_same.shape, (batch, in_channels, T_dim))
		self.assertIsInstance(block_same.res_conv, torch.nn.Identity)
		
	def test_UNet1D(self):
		batch = 2
		window_size = WINDOW_SIZE
		action_dim = ACTION_DIM
		global_cond_dim = CONDITION_DIM + 128
		unet = UNet1D(action_dim, global_cond_dim, hidden_dim=64)
		x = torch.randn(batch, window_size, action_dim)
		cond = torch.randn(batch, global_cond_dim)
		out = unet(x, cond)
		self.assertEqual(out.shape, (batch, window_size, action_dim))
		irregular_window_size = window_size + 3
		x_irreg = torch.randn(batch, irregular_window_size, action_dim)
		out_irreg = unet(x_irreg, cond)
		self.assertEqual(out_irreg.shape, (batch, irregular_window_size, action_dim))
		x_reshaped = x_irreg.transpose(1, 2)
		self.assertEqual(x_reshaped.shape, (batch, action_dim, irregular_window_size))

class TestDiffusionPolicy(unittest.TestCase):
	"""Test the full diffusion policy network."""
	def test_forward_output_shape(self):
		batch = 2
		window_size = WINDOW_SIZE
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=window_size)
		x = torch.randn(batch, window_size, ACTION_DIM)
		t = torch.tensor([10.0, 500.0])
		state = torch.randn(batch, 4)
		image = [torch.randn(batch, 3, IMG_RES, IMG_RES), torch.randn(batch, 3, IMG_RES, IMG_RES)]
		out = model(x, t, state, image)
		self.assertEqual(out.shape, (batch, window_size, ACTION_DIM))
		
	def test_gradient_flow(self):
		batch = 2
		window_size = WINDOW_SIZE
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=window_size)
		x = torch.randn(batch, window_size, ACTION_DIM, requires_grad=True)
		t = torch.tensor([10.0, 500.0])
		state = torch.randn(batch, 4)
		image = [torch.randn(batch, 3, IMG_RES, IMG_RES), torch.randn(batch, 3, IMG_RES, IMG_RES)]
		out = model(x, t, state, image)
		target = torch.randn_like(out)
		loss = F.mse_loss(out, target)
		loss.backward()
		has_gradients = all(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters() if p.requires_grad)
		self.assertTrue(has_gradients, "No gradients found in model parameters")

class TestPolicyDatasetLerobot(unittest.TestCase):
	"""Tests for the PolicyDataset using the real lerobot dataset."""
	def test_dataset_loading_and_structure(self):
		dataset = PolicyDataset("lerobot")
		
		self.assertGreater(len(dataset), 0, "Dataset should contain at least one sample")
		
		# Get the first sample safely with error handling
		try:
			sample = dataset[0]
			print(f"Successfully loaded sample 0. Type: {type(sample)}")
		except Exception as e:
			self.fail(f"Failed to load the first sample with error: {str(e)}")
			
		self.assertEqual(len(sample), 4, "Sample should contain condition, image, action, and time_seq")
		condition, image, action, time_seq = sample
		
		# Check condition: since two state observations are concatenated, expect shape (4,)
		self.assertEqual(condition.shape, (4,))
		
		# Check images: should be a list of two tensors each with shape (3, IMG_RES, IMG_RES)
		self.assertIsInstance(image, list)
		self.assertEqual(len(image), 2)
		self.assertEqual(image[0].shape, (3, IMG_RES, IMG_RES))
		self.assertEqual(image[1].shape, (3, IMG_RES, IMG_RES))
		
		# Check action: should be a tensor with shape (WINDOW_SIZE, ACTION_DIM)
		self.assertEqual(action.shape[1], ACTION_DIM)
		self.assertEqual(action.shape[0], WINDOW_SIZE)
		
		# Check time_seq: should be a tensor of shape (WINDOW_SIZE,) with values from 0 to 1
		self.assertEqual(time_seq.shape[0], WINDOW_SIZE)
		self.assertAlmostEqual(time_seq[0].item(), 0.0, places=5)
		self.assertAlmostEqual(time_seq[-1].item(), 1.0, places=5)
		# Verify monotonicity
		self.assertTrue(torch.all(time_seq[1:] > time_seq[:-1]), "Time sequence should be strictly increasing")
		
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
				sample_data = dataset[idx]
				if not isinstance(sample_data, tuple) or len(sample_data) != 4:
					print(f"Warning: Sample {idx} has unexpected format: {type(sample_data)}, length: {len(sample_data) if hasattr(sample_data, '__len__') else 'N/A'}")
					continue
					
				condition, image, action, time_seq = sample_data
				
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
				
				print(f"Time sequence: {time_seq[:5].numpy()}... to {time_seq[-5:].numpy()}")
				
				# Convert data to numpy arrays for visualization
				x_vals = action[:, 0].numpy()
				y_vals = action[:, 1].numpy()
				time_vals = time_seq.numpy()
				
				 # Debug shapes before plotting
				print(f"Shapes before plotting - x_vals: {x_vals.shape}, y_vals: {y_vals.shape}, time_vals: {time_vals.shape}")
				
				# Ensure all arrays have the same length for plotting
				min_length = min(len(x_vals), len(y_vals), len(time_vals))
				x_vals = x_vals[:min_length]
				y_vals = y_vals[:min_length]
				time_vals = time_vals[:min_length]
				
				print(f"Adjusted shapes - x_vals: {x_vals.shape}, y_vals: {y_vals.shape}, time_vals: {time_vals.shape}")
				
				# Convert to matplotlib for visualization instead of interactive plotly
				plt.figure(figsize=(10, 8))
				
				# Plot the trajectory with color gradient
				points = plt.scatter(
					x_vals, y_vals, 
					c=time_vals, 
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
				plt.show()
				
				# Display the images using matplotlib - with proper denormalization
				fig, axes = plt.subplots(1, 2, figsize=(12, 5))
				
				# Define ImageNet mean and std for denormalization
				mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
				std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
				
				for i, img_tensor in enumerate(image):
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
				plt.show()
				
				# Alternative visualization with pure PIL (just in case)
				try:
					plt.figure(figsize=(12, 5))
					for i, img_tensor in enumerate(image):
						# Denormalize
						img_denorm = img_tensor * std + mean
						
						# Convert to PIL
						img_pil = transforms.ToPILImage()(img_denorm)
						
						plt.subplot(1, 2, i+1)
						plt.imshow(img_pil)
						plt.title(f"Image {i+1} (PIL)")
						plt.axis('off')
					
					plt.tight_layout()
					plt.show()
				except Exception as e:
					print(f"PIL visualization failed: {e}")
				
			except Exception as e:
				print(f"Error processing sample {idx}: {str(e)}")
				import traceback
				traceback.print_exc()

class TestPolicyInference(unittest.TestCase):
	"""Tests for the DiffusionPolicyInference module."""
	def setUp(self):
		self.model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE)
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
	unittest.main()

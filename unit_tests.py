"""
Unit tests for the diffusion policy project.
Tests cover configuration, utility functions, network components,
the complete diffusion policy network, visual encoder, inference module, and dataset.
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
from config import SCREEN_WIDTH, SCREEN_HEIGHT, ARM_LENGTH, BASE_POS, WINDOW_SIZE, ACTION_DIM, CONDITION_DIM, IMG_RES, TRAINING_DATA_DIR, T, IMAGE_FEATURE_DIM
from diffusion_utils import get_beta_schedule, compute_alphas
from diffusion_policy import DiffusionPolicy, UNet1D, ResidualBlock1D, FiLM
from visual_encoder import VisualEncoder
from policy_inference import DiffusionPolicyInference
from train_diffusion import PolicyDataset
from normalize import Normalize
from simulation import *
from utils import *
import tempfile
import torch.nn.functional as F

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
		# Check that the computation matches the expected formula
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
		normalized = self.normalize.normalize_condition(condition)
		self.assertTrue(torch.all(normalized >= -1.0))
		self.assertTrue(torch.all(normalized <= 1.0))
		
	def test_edge_cases(self):
		# Test min values
		min_action = torch.zeros(2, dtype=torch.float32)
		normalized = self.normalize.normalize_action(min_action)
		self.assertTrue(torch.allclose(normalized, torch.tensor([-1.0, -1.0], dtype=torch.float32)))
		
		# Test max values
		max_action = torch.tensor([ACTION_LIM, ACTION_LIM], dtype=torch.float32)
		normalized = self.normalize.normalize_action(max_action)
		self.assertTrue(torch.allclose(normalized, torch.tensor([1.0, 1.0], dtype=torch.float32)))
		
		# Test beyond range (should still normalize correctly)
		beyond_action = torch.tensor([ACTION_LIM * 1.5, ACTION_LIM * 1.5], dtype=torch.float32)
		normalized = self.normalize.normalize_action(beyond_action)
		self.assertTrue(torch.all(normalized > 1.0))  # Should be beyond normalized range
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
		
		# Test that FiLM modulates the features correctly
		# Should apply per-channel bias
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
		
		# Verify the residual connection logic
		# If in_channels != out_channels, should use a 1x1 conv
		self.assertIsInstance(block.res_conv, torch.nn.Conv1d)
		
		# Test with same in and out channels
		block = ResidualBlock1D(in_channels, in_channels, cond_dim)
		out = block(x, cond)
		self.assertEqual(out.shape, (batch, in_channels, T_dim))
		# Should use Identity for same in/out channels
		self.assertIsInstance(block.res_conv, torch.nn.Identity)
		
	def test_UNet1D(self):
		batch = 2
		window_size = WINDOW_SIZE + 1
		action_dim = ACTION_DIM
		global_cond_dim = CONDITION_DIM + 128  # 132 + 128 = 260
		unet = UNet1D(action_dim, global_cond_dim, hidden_dim=64)
		x = torch.randn(batch, window_size, action_dim)
		cond = torch.randn(batch, global_cond_dim)
		out = unet(x, cond)
		self.assertEqual(out.shape, (batch, window_size, action_dim))
		
		# Test with different sequence lengths to verify interpolation
		irregular_window_size = window_size + 3
		x = torch.randn(batch, irregular_window_size, action_dim)
		out = unet(x, cond)
		self.assertEqual(out.shape, (batch, irregular_window_size, action_dim))
		
		# Verify input transformation (reshaping)
		x_reshaped = x.transpose(1, 2)  # Original reshape in forward: b t a -> b a t
		self.assertEqual(x_reshaped.shape, (batch, action_dim, irregular_window_size))

class TestDiffusionPolicy(unittest.TestCase):
	"""Test the full diffusion policy network."""
	def test_forward_output_shape(self):
		batch = 2
		window_size = WINDOW_SIZE + 1
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=window_size)
		x = torch.randn(batch, window_size, ACTION_DIM)
		t = torch.tensor([10.0, 500.0])  # Multiple timesteps
		state = torch.randn(batch, 4)  # Condition state
		# Provide two images as a list
		image = [torch.randn(batch, 3, IMG_RES, IMG_RES), torch.randn(batch, 3, IMG_RES, IMG_RES)]
		out = model(x, t, state, image)
		self.assertEqual(out.shape, (batch, window_size, ACTION_DIM))
		
	def test_gradient_flow(self):
		batch = 2
		window_size = WINDOW_SIZE + 1
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=window_size)
		x = torch.randn(batch, window_size, ACTION_DIM, requires_grad=True)
		t = torch.tensor([10.0, 500.0])
		state = torch.randn(batch, 4)
		image = [torch.randn(batch, 3, IMG_RES, IMG_RES), torch.randn(batch, 3, IMG_RES, IMG_RES)]
		
		# Forward pass
		out = model(x, t, state, image)
		
		# Create dummy target and loss
		target = torch.randn_like(out)
		loss = F.mse_loss(out, target)
		
		# Backprop
		loss.backward()
		
		# Check if gradients exist in the model parameters
		has_gradients = all(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters() if p.requires_grad)
		self.assertTrue(has_gradients, "No gradients found in model parameters")

class TestVisualEncoder(unittest.TestCase):
	"""Tests for the VisualEncoder."""
	def test_forward_output_shape(self):
		encoder = VisualEncoder()
		batch = 2
		x = torch.randn(batch, 3, IMG_RES, IMG_RES)
		features = encoder(x)
		# Expected output shape is (B, 64)
		self.assertEqual(features.shape, (batch, IMAGE_FEATURE_DIM))
		self.assertTrue(torch.isfinite(features).all())
		
	def test_spatial_softmax_properties(self):
		encoder = VisualEncoder()
		batch = 2
		x = torch.randn(batch, 3, IMG_RES, IMG_RES)
		features = encoder(x)
		
		# Test feature bounds - spatial softmax produces coordinates in [-1, 1] range
		self.assertTrue(torch.all(features >= -1.0))
		self.assertTrue(torch.all(features <= 1.0))
		
		# Test that features change when the input changes
		x2 = x + 0.1 * torch.randn_like(x)  # Slightly perturbed input
		features2 = encoder(x2)
		self.assertFalse(torch.allclose(features, features2))
		
		# Test consistency with two identical inputs
		features3 = encoder(x)
		self.assertTrue(torch.allclose(features, features3))

class TestPolicyInference(unittest.TestCase):
	"""Tests for the DiffusionPolicyInference module."""
	def setUp(self):
		# Create a simple model for testing
		self.model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=WINDOW_SIZE+1)
		self.inference = DiffusionPolicyInference(model_path="")
		self.inference.model = self.model
		# We don't need to load a real model for testing
		
	def test_sample_action(self):
		state = torch.randn(1, 4)  # Single batch
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]
		action = self.inference.sample_action(state, image, num_ddim_steps=10)
		self.assertEqual(action.shape, (WINDOW_SIZE, ACTION_DIM))
		self.assertFalse(torch.isnan(action).any())
		
		# Test action clamping - actions should be within screen bounds
		self.assertTrue(torch.all(action[:, 0] >= 0))
		self.assertTrue(torch.all(action[:, 0] <= SCREEN_WIDTH - 1))
		self.assertTrue(torch.all(action[:, 1] >= 0))
		self.assertTrue(torch.all(action[:, 1] <= SCREEN_HEIGHT - 1))
		
	def test_deterministic_sampling(self):
		# Test that sampling is deterministic with fixed seed
		torch.manual_seed(42)
		state = torch.randn(1, 4)
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]
		
		torch.manual_seed(42)  # Reset seed to same value
		action1 = self.inference.sample_action(state, image, num_ddim_steps=10)
		
		torch.manual_seed(42)  # Reset seed to same value again
		action2 = self.inference.sample_action(state, image, num_ddim_steps=10)
		
		self.assertTrue(torch.allclose(action1, action2))
		
	def test_ddim_sampling_time_steps(self):
		# Test with different numbers of sampling steps
		state = torch.randn(1, 4)
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]
		
		# Should work with fewer steps (faster sampling)
		action_fast = self.inference.sample_action(state, image, num_ddim_steps=5)
		self.assertEqual(action_fast.shape, (WINDOW_SIZE, ACTION_DIM))
		
		# Should work with more steps (higher quality)
		action_quality = self.inference.sample_action(state, image, num_ddim_steps=20)
		self.assertEqual(action_quality.shape, (WINDOW_SIZE, ACTION_DIM))

class TestDiffusionProcess(unittest.TestCase):
	"""Tests for the diffusion process (forward and reverse)."""
	def test_forward_diffusion(self):
		# Test the forward diffusion process: q(x_t | x_0)
		batch = 2
		window_size = WINDOW_SIZE + 1
		x_0 = torch.randn(batch, window_size, ACTION_DIM)  # Clean data
		
		# Generate betas and alphas
		betas = get_beta_schedule(T)
		alphas, alphas_cumprod = compute_alphas(betas)
		
		# Sample a timestep
		t = torch.tensor([T//2, T//4])  # Different timesteps for each batch element
		alpha_bar = alphas_cumprod[t].view(batch, 1, 1)
		
		# Generate noise
		torch.manual_seed(0)
		noise = torch.randn_like(x_0)
		
		# Forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
		x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
		
		# Verify shape is preserved
		self.assertEqual(x_t.shape, x_0.shape)
		
		# Verify the signal gets noisier as t increases
		# Higher t should have lower SNR
		snr_0 = torch.mean(torch.abs(torch.sqrt(alpha_bar[0]) * x_0[0]) / 
						  torch.abs(torch.sqrt(1 - alpha_bar[0]) * noise[0]))
		snr_1 = torch.mean(torch.abs(torch.sqrt(alpha_bar[1]) * x_0[1]) / 
						  torch.abs(torch.sqrt(1 - alpha_bar[1]) * noise[1]))
		
		# T//2 > T//4, so SNR should be lower for the first element
		self.assertLess(snr_0, snr_1)

class TestPolicyTraining(unittest.TestCase):
	"""Tests for the policy training components."""
	def setUp(self):
		# Create dummy dataset directory
		self.test_dir = tempfile.mkdtemp()
		self.images_dir = os.path.join(self.test_dir, "images")
		os.makedirs(self.images_dir, exist_ok=True)
		
		# Create dummy images
		for img_name in ["dummy1.png", "dummy2.png"]:
			img_path = os.path.join(self.images_dir, img_name)
			dummy_img = Image.new("RGB", (IMG_RES, IMG_RES), color=(255, 0, 0))
			dummy_img.save(img_path)
		
		# Create dummy data for multiple episodes
		self.dummy_data = []
		for ep_idx in range(3):  # Create 3 episodes
			episode_data = []
			for frame_idx in range(30):  # 30 frames per episode
				sample = {
					"observation.state": [100.0 + frame_idx, 200.0 + frame_idx],
					"action": [300.0, 300.0],
					"episode_index": ep_idx,
					"frame_index": frame_idx,
					"timestamp": 0.1 * frame_idx,
					"index": ep_idx * 30 + frame_idx,
					"task_index": 0,
					"next": {"reward": 1.0, "done": False, "success": False}
				}
				# Mark last frame as done
				if frame_idx == 29:
					sample["next"]["done"] = True
					sample["next"]["success"] = True
				
				episode_data.append(sample)
			
			self.dummy_data.extend(episode_data)
		
		# Save the dummy data
		json_path = os.path.join(self.test_dir, "dummy.json")
		with open(json_path, "w") as f:
			json.dump(self.dummy_data, f)
	
	def tearDown(self):
		shutil.rmtree(self.test_dir)
	
	def test_dataset_chunking(self):
		# Test that dataset properly chunks episodes into training examples
		from train_diffusion import PolicyDataset
		
		original_training_data_dir = TRAINING_DATA_DIR
		try:
			import train_diffusion
			train_diffusion.TRAINING_DATA_DIR = self.test_dir
			
			# Patch the dataset to use our test dir
			dataset = PolicyDataset(self.test_dir)
			
			# Should create multiple chunks from our 3 episodes
			expected_min_chunks = 3 * (30 - WINDOW_SIZE - 1)
			self.assertGreaterEqual(len(dataset), expected_min_chunks)
			
			# Check a sample
			if len(dataset) > 0:
				sample = dataset[0]
				self.assertEqual(len(sample), 4)  # condition, image, action, time_seq
				condition, image, action, time_seq = sample
				
				# Check shapes
				self.assertEqual(condition.shape, (4,))  # Two states flattened
				self.assertEqual(len(image), 2)  # Two images
				self.assertEqual(image[0].shape, (3, IMG_RES, IMG_RES))
				self.assertEqual(action.shape[1], ACTION_DIM)  # Action dimension
				self.assertGreaterEqual(action.shape[0], WINDOW_SIZE)  # Sequence length
		finally:
			train_diffusion.TRAINING_DATA_DIR = original_training_data_dir

	def test_data_time_encoding(self):
		# Test the time encoding for sequences
		from train_diffusion import get_chunk_time_encoding
		
		# Test with different lengths
		for length in [10, 15, 20]:
			time_encoding = get_chunk_time_encoding(length)
			self.assertEqual(time_encoding.shape, (length,))
			self.assertEqual(time_encoding[0].item(), 0.0)  # First value should be 0
			self.assertEqual(time_encoding[-1].item(), 1.0)  # Last value should be 1
			
			# Should be monotonically increasing
			self.assertTrue(torch.all(time_encoding[1:] > time_encoding[:-1]))

if __name__ == '__main__':
	unittest.main()

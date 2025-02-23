import os
import json
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import modules from your project.
from config import SCREEN_WIDTH, SCREEN_HEIGHT, ARM_LENGTH, BASE_POS, WINDOW_SIZE, ACTION_DIM, CONDITION_DIM, IMG_RES, TRAINING_DATA_DIR, T
from diffusion_utils import get_beta_schedule, compute_alphas
from diffusion_policy import DiffusionPolicy, UNet1D, DownBlock, UpBlock, FiLMBlock
from visual_encoder import VisualEncoder
from policy_inference import DiffusionPolicyInference
from train_diffusion import PolicyDataset
from simulation import Simulation

# ---------- Tests for config.py ----------
class TestConfig(unittest.TestCase):
	"""Tests for verifying configuration constants."""

	def test_constants(self):
		"""Test that key configuration constants are of correct type and have expected values."""
		self.assertIsInstance(SCREEN_WIDTH, int)
		self.assertIsInstance(SCREEN_HEIGHT, int)
		self.assertTrue(ARM_LENGTH > 0)
		self.assertTrue(isinstance(BASE_POS, torch.Tensor))
		self.assertEqual(ACTION_DIM, 2)
		self.assertEqual(CONDITION_DIM, 4 + 32)  # 4 state + 32 image features

# ---------- Tests for diffusion_utils.py ----------
class TestDiffusionUtils(unittest.TestCase):
	"""Tests for diffusion utility functions."""

	def test_get_beta_schedule(self):
		"""Verify that the beta schedule is generated with T elements and all values are non-negative."""
		betas = get_beta_schedule(T)
		self.assertEqual(betas.shape[0], T)
		self.assertTrue(torch.all(betas >= 0))

	def test_compute_alphas(self):
		"""Check that alphas are computed as 1 - beta and that cumulative products have correct shape."""
		betas = get_beta_schedule(T)
		alphas, alphas_cumprod = compute_alphas(betas)
		self.assertEqual(alphas.shape[0], T)
		self.assertEqual(alphas_cumprod.shape[0], T)
		self.assertTrue(torch.allclose(alphas, 1 - betas))

# ---------- Tests for diffusion_policy.py components ----------
class TestDiffusionPolicyComponents(unittest.TestCase):
	"""Tests for individual components of the diffusion policy network."""

	def test_FiLMBlock(self):
		"""Test that FiLMBlock produces output of the same shape as input."""
		batch, channels, T_dim = 2, 8, 10
		cond_dim = 16
		film = FiLMBlock(channels, cond_dim)
		x = torch.randn(batch, channels, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = film(x, cond)
		self.assertEqual(out.shape, (batch, channels, T_dim))

	def test_DownBlock(self):
		"""Test that DownBlock correctly downsamples the input and applies FiLM conditioning."""
		batch, in_channels, T_dim = 2, 8, 10
		out_channels = 16
		cond_dim = 12
		block = DownBlock(in_channels, out_channels, cond_dim)
		x = torch.randn(batch, in_channels, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = block(x, cond)
		self.assertEqual(out.shape, (batch, out_channels, T_dim))

	def test_UpBlock(self):
		"""Test that UpBlock correctly upsamples and concatenates skip connections with proper conditioning."""
		batch, channels, T_dim = 2, 16, 10
		out_channels = 8
		cond_dim = 12
		block = UpBlock(channels, out_channels, cond_dim)
		# Create two inputs for concatenation: upsampled x and skip connection.
		x = torch.randn(batch, channels // 2, T_dim)
		skip = torch.randn(batch, channels // 2, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = block(x, skip, cond)
		self.assertEqual(out.shape, (batch, out_channels, T_dim))

	def test_UNet1D(self):
		"""Test that the 1D U-Net returns an output with the correct shape given input and conditioning."""
		batch = 2
		window_size = WINDOW_SIZE + 1
		action_dim = ACTION_DIM
		global_cond_dim = CONDITION_DIM + 128  # state (4) + image features (32) + time embedding (128)
		unet = UNet1D(action_dim, global_cond_dim, hidden_dim=64)
		x = torch.randn(batch, window_size, action_dim)
		cond = torch.randn(batch, global_cond_dim)
		out = unet(x, cond)
		self.assertEqual(out.shape, (batch, window_size, action_dim))

# ---------- Test for DiffusionPolicy (full model) ----------
class TestDiffusionPolicy(unittest.TestCase):
	"""Tests for the complete diffusion policy network."""

	def test_forward_output_shape(self):
		"""Ensure that the diffusion policy forward pass returns an output of the correct shape."""
		batch = 1
		window_size = WINDOW_SIZE + 1
		action_dim = ACTION_DIM
		cond_dim = CONDITION_DIM  # state+image features dimension before time embedding is added.
		model = DiffusionPolicy(action_dim, cond_dim, time_embed_dim=128, window_size=window_size)
		x = torch.randn(batch, window_size, action_dim)
		t = torch.tensor([10.0])  # dummy timestep
		# Create dummy state and dummy image tensor.
		state = torch.randn(batch, 4)
		image = torch.randn(batch, 3, IMG_RES, IMG_RES)
		out = model(x, t, state, image)
		self.assertEqual(out.shape, (batch, window_size, action_dim))

# ---------- Tests for VisualEncoder ----------
class TestVisualEncoder(unittest.TestCase):
	"""Tests for the visual encoder that extracts image features using ResNet18 and spatial softmax."""

	def test_forward_output_shape(self):
		"""Check that VisualEncoder returns a feature vector of shape (B, 32) and that all values are finite."""
		encoder = VisualEncoder()
		batch = 2
		x = torch.randn(batch, 3, IMG_RES, IMG_RES)
		features = encoder(x)
		self.assertEqual(features.shape, (batch, 32))
		self.assertTrue(torch.isfinite(features).all())

# ---------- Tests for PolicyInference ----------
class TestPolicyInference(unittest.TestCase):
	"""Tests for the diffusion policy inference module using DDIM sampling."""

	def setUp(self):
		"""Set up a dummy DiffusionPolicyInference instance and patch the model to avoid instability."""
		self.inference = DiffusionPolicyInference()
		# Monkey-patch the model to a dummy model that returns zeros (to avoid NaNs)
		self.inference.model = lambda x, t, state, image: torch.zeros_like(x)
		# Override T and alphas to simple values.
		self.inference.T = 1000
		self.inference.alphas = torch.linspace(0.9, 0.99, self.inference.T)
		self.inference.alphas_cumprod = torch.cumprod(self.inference.alphas, dim=0)
		self.inference.betas = 1 - self.inference.alphas

	def test_sample_action(self):
		"""Test that sample_action returns a valid target of shape (ACTION_DIM,) with no NaNs."""
		state = torch.randn(1, 4)
		image = torch.randn(1, 3, IMG_RES, IMG_RES)
		action = self.inference.sample_action(state, image, num_ddim_steps=10)
		self.assertEqual(action.shape, (ACTION_DIM,))
		self.assertFalse(torch.isnan(action).any())

# ---------- Tests for PolicyDataset in train_diffusion.py ----------
class TestPolicyDataset(unittest.TestCase):
	"""Tests for the PolicyDataset class to ensure proper data loading and formatting."""

	def setUp(self):
		"""Set up a temporary directory with dummy JSON and image files for dataset testing."""
		self.test_dir = tempfile.mkdtemp()
		self.images_dir = os.path.join(self.test_dir, "images")
		os.makedirs(self.images_dir, exist_ok=True)

		# Create dummy image files.
		for img_name in ["dummy1.png", "dummy2.png"]:
			img_path = os.path.join(self.images_dir, img_name)
			dummy_img = Image.new("RGB", (IMG_RES, IMG_RES), color=(255, 0, 0))
			dummy_img.save(img_path)

		# Create a dummy JSON file with required fields.
		self.dummy_data = [{
			"observation": {
				"state": [[100.0, 200.0], [150.0, 250.0]],
				"image": ["dummy1.png", "dummy2.png"]
			},
			"action": [[300.0, 300.0]] * (WINDOW_SIZE + 15)  # dummy sequence of actions
		}]
		json_path = os.path.join(self.test_dir, "dummy.json")
		with open(json_path, "w") as f:
			json.dump(self.dummy_data, f)

	def tearDown(self):
		"""Remove the temporary directory after tests complete."""
		shutil.rmtree(self.test_dir)

	def test_dataset_getitem(self):
		"""Test that PolicyDataset.__getitem__ correctly loads state, image, and action from a sample."""
		# Temporarily override TRAINING_DATA_DIR to point to self.test_dir.
		original_dir = TRAINING_DATA_DIR
		try:
			import train_diffusion
			train_diffusion.TRAINING_DATA_DIR = self.test_dir

			dataset = PolicyDataset(self.test_dir)
			state, image, action = dataset[0]
			self.assertEqual(state.shape, (4,))
			self.assertEqual(image.shape, (3, IMG_RES, IMG_RES))
			self.assertTrue(action.ndim in (2, 3))
		finally:
			train_diffusion.TRAINING_DATA_DIR = original_dir

# ---------- Tests for Simulation methods ----------
class TestSimulationMethods(unittest.TestCase):
	"""Tests for helper methods in the Simulation class."""

	def setUp(self):
		"""Create a Simulation instance in collection mode for testing."""
		self.sim = Simulation(mode="collection")
		self.sim.prev_ee_pos = torch.tensor([290.0, 290.0], dtype=torch.float32)

	def test_compute_ee_velocity(self):
		"""Test that compute_ee_velocity correctly computes the velocity given current and previous EE positions."""
		current = torch.tensor([300.0, 300.0], dtype=torch.float32)
		dt = 0.01
		vel = Simulation.compute_ee_velocity(current, self.sim.prev_ee_pos, dt)
		expected = (current - self.sim.prev_ee_pos) / dt
		self.assertTrue(torch.allclose(vel, expected))

	def test_update_smoothed_target(self):
		"""Test that update_smoothed_target returns a valid smoothed target without abrupt changes."""
		target = torch.tensor([320.0, 320.0], dtype=torch.float32)
		new_target = self.sim.update_smoothed_target(target)
		self.assertTrue(torch.allclose(new_target, target))
		self.sim.smoothed_target = torch.tensor([310.0, 310.0], dtype=torch.float32)
		updated = self.sim.update_smoothed_target(target)
		self.assertTrue(torch.isfinite(updated).all())

	def test_start_session(self):
		"""Test that start_session initializes the simulation correctly and resets necessary variables."""
		self.sim.start_session()
		self.assertTrue(self.sim.session_active)
		self.assertEqual(self.sim.demo_data, [])
		self.assertIsNone(self.sim.smoothed_target)
		self.assertIsNone(self.sim.prev_ee_pos)

if __name__ == '__main__':
	unittest.main()

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
from diffusion_policy import DiffusionPolicy, UNet1D, DownBlock, UpBlock, FiLMBlock
from visual_encoder import VisualEncoder
from policy_inference import DiffusionPolicyInference
from train_diffusion import PolicyDataset
from simulation import *
from utils import *
import tempfile

class TestConfig(unittest.TestCase):
	"""Tests for configuration constants."""
	def test_constants(self):
		self.assertIsInstance(SCREEN_WIDTH, int)
		self.assertIsInstance(SCREEN_HEIGHT, int)
		self.assertTrue(ARM_LENGTH > 0)
		self.assertIsInstance(BASE_POS, torch.Tensor)
		self.assertEqual(ACTION_DIM, 2)
		self.assertEqual(CONDITION_DIM, 4 + 2 * IMAGE_FEATURE_DIM)  # 4 + 128

class TestDiffusionUtils(unittest.TestCase):
	"""Tests for diffusion utility functions."""
	def test_get_beta_schedule(self):
		betas = get_beta_schedule(T)
		self.assertEqual(betas.shape[0], T)
		self.assertTrue(torch.all(betas >= 0))
	def test_compute_alphas(self):
		betas = get_beta_schedule(T)
		alphas, alphas_cumprod = compute_alphas(betas)
		self.assertEqual(alphas.shape[0], T)
		self.assertEqual(alphas_cumprod.shape[0], T)
		self.assertTrue(torch.allclose(alphas, 1 - betas))

class TestDiffusionPolicyComponents(unittest.TestCase):
	"""Tests for individual network components."""
	def test_FiLMBlock(self):
		batch, channels, T_dim = 2, 8, 10
		cond_dim = 16
		film = FiLMBlock(channels, cond_dim)
		x = torch.randn(batch, channels, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = film(x, cond)
		self.assertEqual(out.shape, (batch, channels, T_dim))
	def test_DownBlock(self):
		batch, in_channels, T_dim = 2, 8, 10
		out_channels = 16
		cond_dim = 12
		block = DownBlock(in_channels, out_channels, cond_dim)
		x = torch.randn(batch, in_channels, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = block(x, cond)
		self.assertEqual(out.shape, (batch, out_channels, T_dim))
	def test_UpBlock(self):
		batch, channels, T_dim = 2, 16, 10
		out_channels = 8
		cond_dim = 12
		block = UpBlock(channels, out_channels, cond_dim)
		x = torch.randn(batch, channels // 2, T_dim)
		skip = torch.randn(batch, channels // 2, T_dim)
		cond = torch.randn(batch, cond_dim)
		out = block(x, skip, cond)
		self.assertEqual(out.shape, (batch, out_channels, T_dim))
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

class TestDiffusionPolicy(unittest.TestCase):
	"""Test the full diffusion policy network."""
	def test_forward_output_shape(self):
		batch = 1
		window_size = WINDOW_SIZE + 1
		model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM, time_embed_dim=128, window_size=window_size)
		x = torch.randn(batch, window_size, ACTION_DIM)
		t = torch.tensor([10.0])
		state = torch.randn(batch, 4)
		# Provide two images as a list.
		image = [torch.randn(batch, 3, IMG_RES, IMG_RES), torch.randn(batch, 3, IMG_RES, IMG_RES)]
		out = model(x, t, state, image)
		self.assertEqual(out.shape, (batch, window_size, ACTION_DIM))

class TestVisualEncoder(unittest.TestCase):
	"""Tests for the VisualEncoder."""
	def test_forward_output_shape(self):
		encoder = VisualEncoder()
		batch = 2
		x = torch.randn(batch, 3, IMG_RES, IMG_RES)
		features = encoder(x)
		# Expected output shape is (B, 64)
		self.assertEqual(features.shape, (batch, 64))
		self.assertTrue(torch.isfinite(features).all())

class TestPolicyInference(unittest.TestCase):
	"""Tests for the DiffusionPolicyInference module."""
	def setUp(self):
		self.inference = DiffusionPolicyInference()
		# Patch the model to avoid instability.
		self.inference.model = lambda x, t, state, image: torch.zeros_like(x)
		self.inference.T = 1000
		self.inference.alphas = torch.linspace(0.9, 0.99, self.inference.T)
		self.inference.alphas_cumprod = torch.cumprod(self.inference.alphas, dim=0)
		self.inference.betas = 1 - self.inference.alphas
	def test_sample_action(self):
		state = torch.randn(1, 4)
		image = [torch.randn(1, 3, IMG_RES, IMG_RES), torch.randn(1, 3, IMG_RES, IMG_RES)]
		action = self.inference.sample_action(state, image, num_ddim_steps=10)
		self.assertEqual(action.shape, (WINDOW_SIZE, ACTION_DIM))
		self.assertFalse(torch.isnan(action).any())

class TestPolicyDataset(unittest.TestCase):
	"""Tests for the PolicyDataset in train_diffusion."""
	def setUp(self):
		self.test_dir = tempfile.mkdtemp()
		self.images_dir = os.path.join(self.test_dir, "images")
		os.makedirs(self.images_dir, exist_ok=True)
		for img_name in ["dummy1.png", "dummy2.png"]:
			img_path = os.path.join(self.images_dir, img_name)
			dummy_img = Image.new("RGB", (IMG_RES, IMG_RES), color=(255, 0, 0))
			dummy_img.save(img_path)
		self.dummy_data = [{
			"observation": {
				"state": [[100.0, 200.0], [150.0, 250.0]],
				"image": ["dummy1.png", "dummy2.png"]
			},
			"action": [[300.0, 300.0]] * (WINDOW_SIZE + 15)
		}]
		json_path = os.path.join(self.test_dir, "dummy.json")
		with open(json_path, "w") as f:
			json.dump(self.dummy_data, f)
	def tearDown(self):
		shutil.rmtree(self.test_dir)
	def test_dataset_getitem(self):
		original_dir = TRAINING_DATA_DIR
		try:
			import train_diffusion
			train_diffusion.TRAINING_DATA_DIR = self.test_dir
			dataset = PolicyDataset(self.test_dir)
			state, image, action, time_seq = dataset[0]
			self.assertEqual(state.shape, (4,))
			self.assertIsInstance(image, (list, tuple))
			self.assertEqual(len(image), 2)
			self.assertEqual(image[0].shape, (3, IMG_RES, IMG_RES))
			self.assertEqual(image[1].shape, (3, IMG_RES, IMG_RES))
			self.assertTrue(action.ndim in (2, 3))
		finally:
			train_diffusion.TRAINING_DATA_DIR = original_dir

if __name__ == '__main__':
	unittest.main()

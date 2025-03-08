"""
Unit test for validating the policy and verifying video saving functionality.

This test creates a model (real if available, mock if not), runs validate_policy with local saving enabled,
and verifies that the video file is correctly saved.
"""

import os
import time
import unittest
import torch
import torch.nn as nn
import numpy as np
from config import *
from diffusion_policy import DiffusionPolicy
from train_diffusion import validate_policy
import cv2


class MockModel(nn.Module):
	"""
	Mock model implementing the required interface for validate_policy.
	"""
	def __init__(self, action_dim=2):
		super().__init__()
		self.action_dim = action_dim
		
	def forward(self, x, t, condition, image):
		# Simple model that returns random actions (for testing only)
		batch_size = x.shape[0]
		return torch.randn_like(x)

	def to(self, device):
		return self


class TestValidation(unittest.TestCase):
	"""
	Test case for validating the policy and video saving functionality.
	"""
	def setUp(self):
		"""
		Set up test environment and mock objects.
		"""
		# Create test directory for videos
		self.test_dir = os.path.join(OUTPUT_DIR, "test_videos")
		os.makedirs(self.test_dir, exist_ok=True)
		
		# Set up a device for testing
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Try to load a real model if available, otherwise use the mock model
		self.model = self.load_real_model() or MockModel()
		
		# Add necessary imports for test
		import importlib
		for module_name in ["gymnasium", "gym_pusht"]:
			try:
				importlib.import_module(module_name)
			except ImportError:
				self.skipTest(f"Required module {module_name} is not available")
				
		# Make sure necessary directories exist
		os.makedirs(os.path.join(OUTPUT_DIR, "lerobot"), exist_ok=True)
		
		# Create a dummy normalization stats file if it doesn't exist
		from normalize import Normalize
		norm_path = OUTPUT_DIR + "lerobot/normalization_stats.parquet"
		if not os.path.exists(norm_path):
			normalize = Normalize.compute_from_limits()
			normalize.save(norm_path)

	def load_real_model(self):
		"""
		Attempt to load a trained model if a checkpoint exists.
		
		Returns:
			DiffusionPolicy or None: Loaded model if successful, None otherwise.
		"""
		# Check common model checkpoint paths
		model_paths = [
			os.path.join(OUTPUT_DIR, "diffusion_policy.pth"),
			os.path.join("lerobot", "diffusion_policy.pth"),
		]
		
		for model_path in model_paths:
			if os.path.exists(model_path):
				print(f"Found model checkpoint at {model_path}")
				try:
					# Initialize the model
					model = DiffusionPolicy(
						action_dim=int(ACTION_DIM),
						condition_dim=int(CONDITION_DIM),
						time_embed_dim=128,
						window_size=int(WINDOW_SIZE + 1)
					).to(self.device)
					
					# Load the checkpoint
					checkpoint = torch.load(model_path, map_location=self.device)
					
					# Handle potential "module." prefix from DataParallel
					new_state_dict = {}
					for key, value in checkpoint.items():
						new_state_dict[key.replace("module.", "")] = value
					
					# Load the state dict
					model.load_state_dict(new_state_dict)
					model.eval()  # Set to evaluation mode
					print(f"Successfully loaded model from {model_path}")
					return model
				except Exception as e:
					print(f"Failed to load model from {model_path}: {e}")
			else:
				print(f"No model checkpoint found at {model_path}")
		
		print("No valid model checkpoint found. Using mock model.")
		return None

	def test_validate_policy_local_save(self):
		"""
		Test that validate_policy can correctly save a video locally.
		"""
		# Define local save path
		timestamp = int(time.time())
		local_save_path = os.path.join(self.test_dir, f"test_video_{timestamp}.avi")
		
		# Ensure wandb doesn't interfere by setting mock run for test
		import wandb
		mock_run_active = hasattr(wandb, 'run') and wandb.run is not None
		print(f"WandB run active before test: {mock_run_active}")
		
		# Report which type of model we're testing with
		if isinstance(self.model, MockModel):
			print("Running test with mock model")
		else:
			print("Running test with real trained model")
		
		try:
			# Run validation with local saving
			reward, video_path = validate_policy(
				model=self.model, 
				device=self.device,
				save_locally=True,
				local_save_path=local_save_path
			)
			
			# Check if a video path was returned
			self.assertIsNotNone(video_path, "Video path should not be None")
			
			# Check if the video file exists
			self.assertTrue(os.path.exists(video_path), f"Video file {video_path} does not exist")
			
			# Check if the video file has content
			self.assertGreater(os.path.getsize(video_path), 0, f"Video file {video_path} is empty")
			
			# Try opening the video to verify it's a valid video file
			cap = cv2.VideoCapture(video_path)
			self.assertTrue(cap.isOpened(), f"Cannot open video file {video_path}")
			
			# Read a frame to verify content
			ret, frame = cap.read()
			self.assertTrue(ret, "Could not read frame from video")
			self.assertIsNotNone(frame, "Frame is None")
			
			# Clean up
			cap.release()
			
			print(f"Test successful: Video saved to {video_path}")
			print(f"Total reward from validation: {reward}")
		
		except Exception as e:
			self.fail(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
	unittest.main()

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
from video_utils import save_video  # Import the new video utility
import cv2
import gymnasium as gym
from PIL import Image
import wandb
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    pass  # Not running on Kaggle


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
		 # Setup WandB login if needed
		self.setup_wandb()
		
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
	
	def setup_wandb(self):
		"""
		Set up WandB login using the same method as in the training script.
		"""
		# Check if wandb is already initialized
		if hasattr(wandb, 'run') and wandb.run is not None:
			print("WandB is already initialized, using existing session")
			return
			
		# Retrieve the WandB API key from the environment variable
		secret_label = "WANDB_API_KEY"
		try:
			api_key = UserSecretsClient().get_secret(secret_label)
		except (KeyError, NameError):
			api_key = None
		
		if api_key is None:
			print("WANDB_API_KEY is not set. Tests that require WandB may fail.")
		else:
			# Log in to WandB using the API key
			try:
				wandb.login(key=api_key)
				print("Successfully logged into WandB")
			except Exception as e:
				print(f"Failed to log into WandB: {e}")
	
	def test_with_wandb_logging(self):
		"""
		Test the validation and video saving with WandB logging enabled.
		"""
		# Initialize a WandB run for testing
		try:
			wandb.init(entity="moribots-personal", project="2d_arm_diffusion", 
					   name="video_logging_test", job_type="testing")
			
			# Create synthetic frames for testing
			frames = self.create_test_frames(30)  # 30 frames
			
			# Test saving with WandB logging
			video_dir = os.path.join(self.test_dir, "wandb_test")
			os.makedirs(video_dir, exist_ok=True)
			
			# Save the video with WandB logging enabled
			video_path, success = save_video(
				frames=frames,
				base_path=video_dir,
				video_identifier="wandb_test",
				wandb_log=True,
				wandb_key="test_video",
				additional_wandb_data={"test_metric": 123.45}
			)
			
			# Check results
			self.assertTrue(success, "Video should save successfully")
			self.assertTrue(os.path.exists(video_path), f"Video file should exist at {video_path}")
			
			# Clean up
			wandb.finish()
			
		except Exception as e:
			if wandb.run is not None:
				wandb.finish()
			self.fail(f"WandB test failed: {e}")

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

	def create_test_frames(self, num_frames=30, width=320, height=240):
		"""
		Create synthetic frames for testing video saving.
		
		Args:
			num_frames: Number of frames to generate
			width: Frame width
			height: Frame height
			
		Returns:
			List of numpy arrays representing the frames
		"""
		frames = []
		for i in range(num_frames):
			# Create a frame with a colored rectangle
			frame = np.zeros((height, width, 3), dtype=np.uint8)
			color = (i * 8 % 256, 255 - (i * 8 % 256), 128)
			top_left = (width//4, height//4)
			bottom_right = (3*width//4, 3*height//4)
			cv2.rectangle(frame, top_left, bottom_right, color, -1)
			
			# Add frame number text
			cv2.putText(frame, f"Frame {i}", (width//4, height//8), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			
			frames.append(frame)
		return frames

	def test_validate_policy_local_save(self):
		"""
		Test that validate_policy can correctly save a video locally.
		"""
		# Define local save path
		timestamp = int(time.time())
		local_save_path = os.path.join(self.test_dir, f"test_video_{timestamp}.mp4")
		
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

	def test_direct_video_saving(self):
		"""
		Test the video saving utility directly with synthetic frames.
		"""
		# Create synthetic frames (colored rectangles)
		frames = []
		width, height = 320, 240
		for i in range(30):  # 30 frames
			# Create a frame with a colored rectangle
			frame = np.zeros((height, width, 3), dtype=np.uint8)
			color = (i * 8 % 256, 255 - (i * 8 % 256), 128)
			top_left = (width//4, height//4)
			bottom_right = (3*width//4, 3*height//4)
			cv2.rectangle(frame, top_left, bottom_right, color, -1)
			
			# Add frame number text
			cv2.putText(frame, f"Frame {i}", (width//4, height//8), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			
			frames.append(frame)
		
		# Test saving with different options
		test_cases = [
			{"name": "standard", "path": os.path.join(self.test_dir, "standard")},
			{"name": "high_fps", "path": os.path.join(self.test_dir, "high_fps"), "fps": 60},
			{"name": "explicit_id", "path": os.path.join(self.test_dir, "explicit_id"), "id": "test_explicit"}
		]
		
		for test_case in test_cases:
			print(f"\nTesting video saving: {test_case['name']}")
			os.makedirs(test_case["path"], exist_ok=True)
			
			kwargs = {
				"frames": frames,
				"base_path": test_case["path"],
				"wandb_log": False  # Don't log to WandB during tests
			}
			
			if "fps" in test_case:
				kwargs["fps"] = test_case["fps"]
			if "id" in test_case:
				kwargs["video_identifier"] = test_case["id"]
			
			video_path, success = save_video(**kwargs)
			
			# Check results
			print(f"Save result: success={success}, path={video_path}")
			
			if success:
				self.assertIsNotNone(video_path)
				self.assertTrue(os.path.exists(video_path))
				self.assertGreater(os.path.getsize(video_path), 0)
				
				# Try opening the video
				cap = cv2.VideoCapture(video_path)
				self.assertTrue(cap.isOpened(), f"Cannot open video file {video_path}")
				
				# Get video properties
				actual_fps = cap.get(cv2.CAP_PROP_FPS)
				frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
					
				# Clean up
				cap.release()
			else:
				# If failed, should not return a path
				self.assertIsNone(video_path)

	def test_gif_saving(self):
		"""
		Test that GIFs can be successfully created and saved.
		"""
		# Create synthetic frames
		frames = self.create_test_frames(30)  # 30 frames
		
		# Try saving as GIF locally for testing
		video_dir = os.path.join(self.test_dir, "gif_test")
		os.makedirs(video_dir, exist_ok=True)
		
		# Save the video with GIF format and explicit save_locally=True for testing
		gif_path, success = save_video(
			frames=frames,
			base_path=video_dir,
			video_identifier="gif_test",
			wandb_log=False,
			use_gif=True,
			save_locally=True  # Explicitly save locally for testing
		)
		
		# Check results
		self.assertTrue(success, "GIF should save successfully")
		self.assertTrue(os.path.exists(gif_path), f"GIF file should exist at {gif_path}")
		self.assertGreater(os.path.getsize(gif_path), 0, "GIF file should have content")
		self.assertEqual(gif_path.endswith('.gif'), True, "File should have .gif extension")
		
		# Test WandB-only saving (mocked)
		from unittest.mock import patch, MagicMock
		
		# Create a mock WandB run
		mock_wandb_run = MagicMock()
		mock_wandb = MagicMock()
		mock_wandb.run = mock_wandb_run
		mock_wandb.Video = MagicMock(return_value="mock_video")
		mock_wandb.log = MagicMock()
		
		with patch.dict('sys.modules', {'wandb': mock_wandb}):
			# Test WandB-only GIF (no local save)
			_, wandb_success = save_video(
				frames=frames,
				base_path=video_dir,
				video_identifier="wandb_test",
				wandb_log=True,
				save_locally=False  # Don't save locally
			)
			
			# Verify WandB would have been called
			self.assertTrue(wandb_success, "WandB-only save should report success")
			mock_wandb.log.assert_called()  # WandB.log should have been called


if __name__ == "__main__":
	unittest.main()

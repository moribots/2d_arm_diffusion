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
import argparse
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
	# Class variable to control WandB logging
	use_wandb = False
	
	def setUp(self):
		"""
		Set up test environment and mock objects.
		"""
		# Setup WandB login if needed
		if self.use_wandb:
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
			api_key = "b716ec3f9f60902ae83d56040350b65a50d616b6"
		
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
		 # Skip this test if WandB is not enabled
		if not self.use_wandb:
			self.skipTest("Skipping WandB test as use_wandb=False")
			return
		
		# Initialize a WandB run for testing
		try:
			wandb.init(entity="moribots-personal", project="2d_arm_diffusion", 
					   name="video_logging_test", job_type="testing")
			
			# Create synthetic frames for testing
			frames = self.create_test_frames(30)  # 30 frames
			
			# Test saving with WandB logging
			video_dir = os.path.join(self.test_dir, "wandb_test")
			os.makedirs(video_dir, exist_ok=True)
			
			# Save the video with WandB logging enabled, but don't save locally
			video_path, success = save_video(
				frames=frames,
				base_path=video_dir,
				video_identifier="wandb_test",
				wandb_log=True,
				wandb_key="test_video",
				additional_wandb_data={"test_metric": 123.45},
				save_locally=False  # Don't save locally
			)
			
			# Check results
			self.assertTrue(success, "WandB logging should succeed")
			
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
		local_save_path = os.path.join(self.test_dir, f"test_video_{timestamp}.gif")
		
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
			# Run validation with local saving - now expecting 3 return values
			reward, video_path, action_plot_path = validate_policy(
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
			
			# If it's a GIF, we don't need to use OpenCV
			if video_path.endswith('.gif'):
				print(f"Test successful: GIF saved to {video_path}")
			else:
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
			
			# Check if an action plot was generated
			if action_plot_path is not None:
				print(f"Action plot generated at: {action_plot_path}")
				
				# Check if the plot file exists
				self.assertTrue(os.path.exists(action_plot_path), f"Plot file {action_plot_path} does not exist")
				
				# Check if the plot file has content
				self.assertGreater(os.path.getsize(action_plot_path), 0, f"Plot file {action_plot_path} is empty")
				
				# If the plot is HTML, try opening it in a browser
				if action_plot_path.endswith('.html'):
					import webbrowser
					try:
						# Try to open the HTML file in a browser
						webbrowser.open('file://' + os.path.abspath(action_plot_path))
						print("Opening action plot in browser...")
					except Exception as e:
						print(f"Could not open plot in browser: {e}")
			else:
				print("No action plot was generated.")
			
			print(f"Total reward from validation: {reward}")
		
		except Exception as e:
			self.fail(f"Test failed with error: {str(e)}")

	def test_direct_video_saving(self):
		"""
		Test the video saving utility directly with synthetic frames.
		"""
		# Create synthetic frames
		frames = self.create_test_frames(30)  # 30 frames
		
		# Test saving with different options, but only to WandB if enabled
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
				"wandb_log": self.use_wandb,  # Only log to WandB if enabled
				"save_locally": False  # Don't save files locally
			}
			
			if "fps" in test_case:
				kwargs["fps"] = test_case["fps"]
			if "id" in test_case:
				kwargs["video_identifier"] = test_case["id"]
			
			# If we're not using WandB and not saving locally, expect failure
			if not self.use_wandb:
				with self.assertRaises(Exception) as context:
					video_path, success = save_video(**kwargs)
				print(f"Expected failure when not saving locally and not using WandB: {context.exception}")
			else:
				video_path, success = save_video(**kwargs)
				self.assertTrue(success, "Should succeed with WandB logging")
				self.assertIsNone(video_path, "No local path should be returned when save_locally=False")

	def test_gif_saving(self):
		"""
		Test that GIFs can be successfully created and saved.
		"""
		# Skip actual file saving and only test the mechanics
		if not self.use_wandb:
			from unittest.mock import patch, MagicMock
			
			# Create synthetic frames
			frames = self.create_test_frames(30)  # 30 frames
			
			# Mock the actual file writing operations
			with patch('cv2.VideoWriter') as mock_video_writer, \
				 patch('PIL.Image.Image.save') as mock_image_save:
				
				# Configure the mocks
				mock_instance = MagicMock()
				mock_video_writer.return_value = mock_instance
				mock_instance.isOpened.return_value = True
				
				# Test the GIF conversion path without saving
				with self.assertRaises(Exception) as context:
					save_video(
						frames=frames,
						base_path=self.test_dir,
						video_identifier="gif_test",
						wandb_log=False,
						use_gif=True,
						save_locally=False  # Don't save locally
					)
				
				# The operation should fail, but we just want to verify the code path
				print(f"Expected failure when not saving locally and not using WandB: {context.exception}")
		else:
			# Create synthetic frames
			frames = self.create_test_frames(30)  # 30 frames
			
			# Test WandB GIF logging without local saving
			gif_path, success = save_video(
				frames=frames,
				base_path=self.test_dir,
				video_identifier="gif_test",
				wandb_log=True,
				use_gif=True,
				save_locally=False  # Don't save locally
			)
			
			# Check results
			self.assertTrue(success, "WandB logging should succeed")
			self.assertIsNone(gif_path, "No local path should be returned when save_locally=False")


if __name__ == "__main__":
	# Add command line argument parser
	parser = argparse.ArgumentParser(description='Run validation tests')
	parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging for tests')
	
	# Parse arguments and set class variables
	args, unknown = parser.parse_known_args()
	TestValidation.use_wandb = args.use_wandb
	
	# Remove the custom arguments to prevent unittest from complaining
	import sys
	sys.argv = [sys.argv[0]] + unknown
	
	# Run the tests
	unittest.main()

#!/usr/bin/env python
"""
This module implements simulation classes for the push environment.
It defines BaseSimulation and specialized simulations for LeRobot and custom environments.
Simulation logs are saved using Parquet.
"""

import argparse
import pygame
import os
import time
import torch
from config import *
from utils import get_training_data_dir, recompute_normalization_stats
from arm import ArmNR
from object import Object
from policy_inference import DiffusionPolicyInference
from einops import rearrange
from PIL import Image
import shutil
import gymnasium as gym
import gym_pusht
import numpy as np
from torchvision.transforms import ToPILImage
import pyarrow.parquet as pq
from pyarrow import Table

class BaseSimulation:
	"""
	BaseSimulation initializes pygame and handles logging simulation steps in a common LeRobot format.
	"""
	def __init__(self, mode="collection", env_type="custom"):
		pygame.init()
		self.mode = mode                # "collection" or "inference"
		self.env_type = env_type        # "custom" or "lerobot"
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		pygame.display.set_caption("Push Simulation")
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("Arial", 20)
		self.episode_index = 0
		self.global_index = 0
		self.task_index = 0
		self.frame_index = 0
		self.episode_start_time = 0.0
		self.le_robot_data = []         # List to store logged steps.
		self.session_active = False
		self.training_data_dir = get_training_data_dir(self.env_type)
		os.makedirs(self.training_data_dir, exist_ok=True)

	def run(self):
		# Must be implemented by subclasses.
		raise NotImplementedError

	def log_data(self, observation_state, action, reward, done, success):
		"""
		Log a single simulation step.
		"""
		step_log = {
			"observation.state": observation_state,
			"action": action,
			"episode_index": self.episode_index,
			"frame_index": self.frame_index,
			"timestamp": time.time() - self.episode_start_time,
			"index": self.global_index,
			"task_index": self.task_index,
			"next": {
				"reward": float(reward),
				"done": bool(done),
				"success": bool(success)
			}
		}
		self.le_robot_data.append(step_log)
		self.frame_index += 1
		self.global_index += 1

	def save_data(self):
		"""
		Save the simulation log to a Parquet file.
		"""
		if not self.le_robot_data:
			print("No data to save.")
			return
		# Create a pyarrow Table from the log dictionary.
		table = Table.from_pydict({k: [step[k] for step in self.le_robot_data] for k in self.le_robot_data[0]})
		filename = os.path.join(self.training_data_dir, f"{self.env_type}_collection_{int(time.time())}.parquet")
		base_filename = filename[:-8]
		counter = 1
		while os.path.exists(filename):
			filename = f"{base_filename}_{counter}.parquet"
			counter += 1
		pq.write_table(table, filename)
		print(f"Data saved to {filename}")
		self.le_robot_data = []

class LeRobotSimulation(BaseSimulation):
	"""
	LeRobotSimulation uses the LeRobot Gym environment.
	In inference mode, it uses a diffusion policy for action selection,
	processing two images (from t-1 and t) for conditioning.
	"""
	def __init__(self, mode="collection"):
		super().__init__(mode=mode, env_type="lerobot")
		self.env = gym.make(LE_ROBOT_GYM_ENV_NAME, obs_type="pixels_agent_pos", render_mode="rgb_array")
		self.env._max_episode_steps = 1e9
		self.policy_inference = None
		self.prev_agent_pos = None
		self.prev_image = None  # Stores previous image (t-1)
		# Remove action buffer tracking - using policy_inference's internal buffering instead
		self.sim_start_time = 0.0  # Track simulation start time for time-based action interpolation
		self.steps = 0
		if self.mode == "inference":
			recompute_normalization_stats(self.env_type, "lerobot/")
			self.policy_inference = DiffusionPolicyInference(model_path="lerobot/diffusion_policy.pth",
															  norm_stats_path="lerobot/normalization_stats.parquet")
			print("LeRobot: Inference mode.")
		else:
			print("LeRobot: Collection mode.")

	def run(self):
		print("Press [N] to start a new LeRobot session. Press [Q] to quit.")
		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					return
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_q:
						pygame.quit()
						return
					if event.key == pygame.K_n:
						if not self.session_active:
							self.start_session()
						else:
							self.end_session(force=True)
			self.screen.fill(BACKGROUND_COLOR)
			if self.session_active:
				self.step_session()
			else:
				text = self.font.render("[N] to start new LeRobot session", True, (0, 0, 0))
				self.screen.blit(text, (20, 20))
			pygame.display.flip()
			self.clock.tick(FPS)

	def start_session(self):
		self.session_active = True
		self.le_robot_data = []
		self.episode_index += 1
		self.frame_index = 0
		self.global_index = 0
		self.episode_start_time = time.time()
		self.prev_agent_pos = None
		self.prev_image = None
		# Reset simulation time tracking
		self.sim_start_time = 0.0
		self.steps = 0
		self.observation, info = self.env.reset()
		print("LeRobot session started.")

	def end_session(self, force=False):
		print("LeRobot session ended.")
		# Commonize: always save log data if in collection mode.
		if self.mode == "collection":
			self.save_data()
		self.session_active = False

	def step_session(self):
		"""
		Executes one simulation step.
		In collection mode, uses mouse input.
		In inference mode, uses the diffusion policy with two images,
		using the policy_inference class for time-based action interpolation.
		"""
		done = False
		truncated = False
		if self.mode == "collection":
			mouse_pos = pygame.mouse.get_pos()
			action = np.array([
				mouse_pos[0] * ACTION_LIM / SCREEN_WIDTH,
				mouse_pos[1] * ACTION_LIM / SCREEN_HEIGHT
			], dtype=np.float32)
		else:
			# Build state from current and previous agent positions
			agent_pos_np = np.array(self.observation["agent_pos"])
			agent_pos_tensor = torch.from_numpy(agent_pos_np).float()
			if self.prev_agent_pos is None:
				state = torch.cat([agent_pos_tensor, agent_pos_tensor], dim=0).unsqueeze(0)
			else:
				state = torch.cat([self.prev_agent_pos.clone(), agent_pos_tensor], dim=0).unsqueeze(0)
			self.prev_agent_pos = agent_pos_tensor.clone()
			
			# Process current image: add batch dimension
			image_array = self.observation["pixels"]
			if not isinstance(image_array, np.ndarray):
				image_array = np.array(image_array, dtype=np.uint8)
			current_image_tensor = image_transform(Image.fromarray(image_array)).unsqueeze(0)
			
			# Use previous image if available; otherwise, duplicate
			if self.prev_image is None:
				image_tuple = [current_image_tensor, current_image_tensor]
			else:
				image_tuple = [self.prev_image, current_image_tensor]
			self.prev_image = current_image_tensor.clone()
			
			# Calculate current simulation time
			current_time = self.sim_start_time + (self.steps + 1) * (1.0 / self.env.metadata['render_fps'])
			
			# Get action from policy, providing current time for proper interpolation
			action = self.policy_inference.sample_action(
				state, 
				image_tuple,
				current_time=current_time
			).cpu().numpy()

			self.steps += 1
		
		# Step environment
		next_observation, reward, done, truncated, info = self.env.step(action)
		obs_list = self.observation["agent_pos"].tolist()
		act_list = action.tolist()
		success = bool(done)
		self.log_data(obs_list, act_list, float(reward), bool(done), success)
		img_arr = self.env.render()
		if img_arr is not None:
			surface = pygame.surfarray.make_surface(img_arr.swapaxes(0, 1))
			self.screen.blit(surface, (0, 0))
		self.observation = next_observation
		if done or truncated:
			self.end_session(force=False)

class CustomSimulation(BaseSimulation):
	"""
	CustomSimulation implements a custom simulation.
	In inference mode, it uses a diffusion policy that processes two images for conditioning.
	"""
	def __init__(self, mode="collection"):
		super().__init__(mode=mode, env_type="custom")
		self.arm = ArmNR(BASE_POS, LINK_LENGTHS, torch.tensor(INITIAL_ANGLES, dtype=torch.float32))
		self.object = Object(random_t_pose())
		self.goal_pose = torch.tensor([212, 413, -math.pi / 3.0], dtype=torch.float32)
		self.policy_inference = None
		self.smoothed_target = None
		self.prev_ee_pos = None
		self.last_target = None
		self.max_target_delta = 10
		self.images_dir = os.path.join(self.training_data_dir, "images")
		os.makedirs(self.images_dir, exist_ok=True)
		self.prev_image = None
		# Add simulation time tracking
		self.sim_start_time = 0.0
		self.steps = 2
		if self.mode == "inference":
			recompute_normalization_stats(self.env_type, "custom/")
			self.policy_inference = DiffusionPolicyInference(model_path="custom/diffusion_policy.pth",
															  norm_stats_path="custom/normalization_stats.parquet")
			print("Custom: Inference mode.")
		else:
			print("Custom: Collection mode.")

	def run(self):
		print("Press [N] to start a new custom session. Press [Q] to quit.")
		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					return
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_q:
						pygame.quit()
						return
					if event.key == pygame.K_n:
						if not self.session_active:
							self.start_session()
						else:
							self.end_session(force=True)
			self.screen.fill(BACKGROUND_COLOR)
			if self.session_active:
				self.step_session()
			else:
				text = self.font.render("[N] to start new custom session", True, (0, 0, 0))
				self.screen.blit(text, (20, 20))
			pygame.display.flip()
			self.clock.tick(FPS)

	def start_session(self):
		# Reset simulation state.
		self.session_active = True
		self.le_robot_data = []
		self.episode_index += 1
		self.frame_index = 0
		self.global_index = 0
		self.episode_start_time = time.time()
		self.object = Object(random_t_pose())
		self.arm.joint_angles = torch.zeros(NUM_JOINTS, dtype=torch.float32)
		self.object.velocity = torch.zeros(2, dtype=torch.float32)
		self.object.angular_velocity = 0.0
		self.smoothed_target = None
		self.prev_ee_pos = None
		self.last_target = None
		self.prev_image = None
		# Reset simulation time tracking
		self.sim_start_time = 0.0
		self.steps = 0
		self.temp_images_dir = os.path.join(self.training_data_dir, f"temp_images_{int(time.time())}")
		os.makedirs(self.temp_images_dir, exist_ok=True)
		print("Custom session started.")

	def end_session(self, force=False):
		print("Custom session ended.")
		# Commonize: always save log data (if in collection mode).
		if self.mode == "collection":
			self.save_data()
		self.clean_up_temp_images()
		self.session_active = False

	def step_session(self):
		dt = 1.0 / FPS
		target = self.get_target_input()
		if torch.norm(target - BASE_POS) > ARM_LENGTH:
			print("Input outside arm workspace => terminating session.")
			self.log_data([0, 0], target.cpu().numpy().tolist(), 0.0, True, False)
			self.end_session(force=True)
			return
		self.smoothed_target = self.update_smoothed_target(target)
		ik_error = self.arm.solve_ik(self.smoothed_target)
		if not torch.isfinite(torch.tensor(ik_error)) or ik_error > 1e4:
			print("Invalid IK => terminating session.")
			self.log_data([0, 0], target.cpu().numpy().tolist(), 0.0, True, False)
			self.end_session(force=True)
			return
		ee_pos = self.arm.forward_kinematics()
		ee_velocity = compute_ee_velocity(ee_pos, self.prev_ee_pos, dt)
		self.prev_ee_pos = ee_pos.clone()
		self.update_object_physics(ee_pos, ee_velocity)
		self.draw_goal_T()
		self.arm.draw(self.screen)
		self.object.draw(self.screen)
		pygame.draw.circle(self.screen, (0, 200, 0), (int(target[0].item()), int(target[1].item())), 6)
		pos_error = torch.norm(self.object.pose[:2] - self.goal_pose[:2]).item()
		orient_error = abs(self.object.pose[2] - self.goal_pose[2])
		done = bool(pos_error < GOAL_POS_TOL and orient_error < GOAL_ORIENT_TOL)
		reward = float(pos_error)
		success = bool(done)
		observation_state = ee_pos.cpu().numpy().tolist()
		action_list = self.last_target.cpu().numpy().tolist() if self.last_target is not None else [0, 0]
		self.log_data(observation_state, action_list, reward, done, success)
		if done:
			print("Goal reached => ending session.")
			self.end_session(force=False)

	def clean_up_temp_images(self):
		if not hasattr(self, "temp_images_dir") or not os.path.exists(self.temp_images_dir):
			return
		for file_name in os.listdir(self.temp_images_dir):
			src = os.path.join(self.temp_images_dir, file_name)
			dst = os.path.join(self.images_dir, file_name)
			shutil.move(src, dst)
		os.rmdir(self.temp_images_dir)

	def get_target_input(self) -> torch.Tensor:
		"""
		Obtain the target input either from the diffusion policy (in inference mode)
		or from mouse input (in collection mode).
		Uses the policy_inference class for time-based action interpolation.
		"""
		if self.mode == "inference":
			current_ee = self.arm.forward_kinematics()
			if self.prev_ee_pos is None:
				state = torch.cat([current_ee, current_ee], dim=0).unsqueeze(0)
			else:
				state = torch.cat([self.prev_ee_pos, current_ee], dim=0).unsqueeze(0)
			
			image = self.get_current_image_tensor(current_ee)
			
			# Calculate current simulation time
			current_time = self.sim_start_time + (self.steps + 1) * (1.0 / FPS)
			
			# Get action using time-based interpolation
			target = self.policy_inference.sample_action(
				state,
				image,
				current_time=current_time
			)
			
			self.last_target = target

			self.steps += 1
			return target
		else:
			mouse_pos = pygame.mouse.get_pos()
			action = np.array([mouse_pos[0], mouse_pos[1]], dtype=np.float32)
			target = torch.tensor(action, dtype=torch.float32)
			self.last_target = target
			return target

	def update_smoothed_target(self, target: torch.Tensor) -> torch.Tensor:
		if self.smoothed_target is None:
			return target.clone()
		delta = target - self.smoothed_target
		delta_clamped = torch.clamp(delta, -self.max_target_delta, self.max_target_delta)
		return self.smoothed_target + delta_clamped

	def draw_goal_T(self):
		world_vertices = Object.get_transformed_polygon(self.goal_pose)
		pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
		pygame.draw.polygon(self.screen, GOAL_T_COLOR, pts, width=3)

	def update_object_physics(self, ee_pos, ee_velocity):
		force, push_direction, dist, contact_pt = self.object.get_contact_force(
			ee_pos, ee_velocity, EE_RADIUS, K_CONTACT, K_DAMPING
		)
		if force is not None:
			torque, true_centroid = self.object.update(
				force, 1.0 / FPS, contact_pt, velocity_force=K_DAMPING * (ee_velocity - self.object.velocity)
			)
			self.draw_contact_forces(force, contact_pt, true_centroid, torque)
		else:
			self.object.velocity = torch.zeros(2, dtype=torch.float32)
			self.object.angular_velocity = 0.0

	def draw_contact_forces(self, force, contact_pt, true_centroid, torque):
		contact_x = int(contact_pt[0].item())
		contact_y = int(contact_pt[1].item())
		cross_size = 5
		pygame.draw.line(self.screen, (255, 0, 0),
						 (contact_x - cross_size, contact_y - cross_size),
						 (contact_x + cross_size, contact_y + cross_size), 2)
		pygame.draw.line(self.screen, (255, 0, 0),
						 (contact_x - cross_size, contact_y + cross_size),
						 (contact_x + cross_size, contact_y - cross_size), 2)
		centroid_pt = (int(true_centroid[0].item()), int(true_centroid[1].item()))
		pygame.draw.line(self.screen, (255, 0, 0), centroid_pt, (contact_x, contact_y), 3)

	def get_current_image_tensor(self, target=None, for_save=False):
		"""
		Capture the current simulation image and process it using the image transform.
		Returns a list of two image tensors (each with batch dimension) for t-1 and t.
		"""
		if target is not None:
			surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
			surface.fill(BACKGROUND_COLOR)
			world_vertices = Object.get_transformed_polygon(self.goal_pose)
			pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
			pygame.draw.polygon(surface, GOAL_T_COLOR, pts, width=3)
			self.object.draw(surface)
			ee = self.arm.forward_kinematics()
			pygame.draw.circle(surface, (0, 0, 0), (int(ee[0].item()), int(ee[1].item())), EE_RADIUS)
			pygame.draw.circle(surface, (0, 200, 0), (int(target[0].item()), int(target[1].item())), 6)
			scaled_surface = pygame.transform.scale(surface, (IMG_RES, IMG_RES))
			raw_str = pygame.image.tostring(scaled_surface, 'RGB')
			pil_img = Image.frombytes('RGB', (IMG_RES, IMG_RES), raw_str)
		else:
			data = pygame.image.tostring(self.screen, 'RGB')
			pil_img = Image.frombytes('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), data)
		current_image_tensor = image_transform(pil_img)
		if current_image_tensor.dim() == 3:
			current_image_tensor = current_image_tensor.unsqueeze(0)
		if self.prev_image is None:
			image_tuple = [current_image_tensor, current_image_tensor]
		else:
			image_tuple = [self.prev_image, current_image_tensor]
		self.prev_image = current_image_tensor.clone()
		if for_save:
			return pil_img
		else:
			return image_tuple

def create_simulation(mode: str, env_type: str):
	"""
	Factory function to create a simulation instance based on the environment type.
	"""
	from simulation import LeRobotSimulation, CustomSimulation  # Local import.
	if env_type == "lerobot":
		return LeRobotSimulation(mode=mode)
	else:
		return CustomSimulation(mode=mode)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run Push Simulation in collection or inference mode.")
	parser.add_argument("--mode", type=str, default="collection", choices=["collection", "inference"], help="Choose mode.")
	parser.add_argument("--env", type=str, default="custom", choices=["custom", "lerobot"], help="Select environment type.")
	args = parser.parse_args()
	sim = create_simulation(args.mode, args.env)
	sim.run()

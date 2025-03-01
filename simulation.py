#!/usr/bin/env python

import argparse
import pygame
import json
import os
import time
import torch
from config import *
from utils import *
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

# -----------------------------------------------------------
# BaseSimulation
# -----------------------------------------------------------
class BaseSimulation:
	"""
	Common parent class for shared logic:
	  - Pygame init
	  - Mode/env fields
	  - Logging methods for LeRobot-format
	"""

	def __init__(self, mode="collection", env_type="custom"):
		self.mode = mode  # "collection" or "inference"
		self.env_type = env_type  # "custom" or "lerobot"

		# Pygame display, font, etc.
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		pygame.display.set_caption("Push Simulation")
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("Arial", 20)

		# Episode / logging state
		self.episode_index = 0
		self.global_index = 0
		self.task_index = 0
		self.frame_index = 0
		self.episode_start_time = 0.0

		# For step-by-step logs in LeRobot format
		self.le_robot_data = []

		# Boolean to indicate if a session is active
		self.session_active = False

		# Prepare directories
		self.training_data_dir = get_training_data_dir(self.env_type)
		os.makedirs(self.training_data_dir, exist_ok=True)

	def run(self):
		"""
		Subclasses must implement run().
		"""
		raise NotImplementedError

	# -------------------------------------------------------
	# Shared logging: store a single step in LeRobot format
	# -------------------------------------------------------
	def log_le_robot_step(self,
						  observation_state,
						  action,
						  reward: float,
						  done: bool,
						  success: bool):
		"""
		Appends a single step to self.le_robot_data in LeRobot format.
		observation_state: Python list or tuple of floats
		action: Python list or tuple of floats
		reward: float (pose error, etc.)
		done: bool
		success: bool
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

		# increment frame & global counters
		self.frame_index += 1
		self.global_index += 1

	# -------------------------------------------------------
	# Save all collected data in LeRobot format, then reset
	# -------------------------------------------------------
	def save_le_robot_data(self):
		"""
		Write the entire step log (self.le_robot_data) to a JSON file
		in the correct directory (custom or lerobot).
		"""
		if not self.le_robot_data:
			print("No data to save.")
			return

		final_dict = {"train": self.le_robot_data}
		filename = os.path.join(self.training_data_dir,
								f"{self.env_type}_collection_{int(time.time())}.json")
		base_filename = filename[:-5]
		counter = 1
		while os.path.exists(filename):
			filename = f"{base_filename}_{counter}.json"
			counter += 1

		with open(filename, "w") as f:
			json.dump(final_dict, f, indent=2)

		print(f"Data saved to {filename}")

		# Clear out the list for the next session
		self.le_robot_data = []


# -----------------------------------------------------------
# LeRobotSimulation
# -----------------------------------------------------------
class LeRobotSimulation(BaseSimulation):
	"""
	LeRobot environment simulation that also waits for [N] to start a session.
	In 'collection' mode: user uses mouse to provide the action.
	In 'inference' mode: uses a diffusion policy.
	"""

	def __init__(self, mode="collection"):
		super().__init__(mode=mode, env_type="lerobot")

		# Initialize the LeRobot Gym environment
		self.env = gym.make(LE_ROBOT_GYM_ENV_NAME, obs_type="pixels_agent_pos", render_mode="rgb_array")
		# Remove or set large to effectively have no step limit
		self.env._max_episode_steps = 1e9

		# Optionally load inference policy
		self.policy_inference = None
		self.prev_agent_pos = None
		if self.mode == "inference":
			# Load existing data if needed to update normalization stats
			recompute_normalization_stats(self.env_type, "lerobot/")
			self.policy_inference = DiffusionPolicyInference(model_path="lerobot/diffusion_policy.pth",
													norm_stats_path="lerobot/normalization_stats.json")
			print("LeRobot: Inference mode.")
		else:
			print("LeRobot: Collection mode.")

	def run(self):
		"""
		Main loop:
		 - Press [N] to start/stop a session.
		 - If a session is active, step the environment each frame.
		"""
		print("Press [N] to start a new push session. Press [Q] to quit.")
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
							# End the session forcibly if desired
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
		"""
		Reset environment, reset logs, mark session as active.
		"""
		self.session_active = True
		self.le_robot_data = []
		self.episode_index += 1
		self.frame_index = 0
		self.global_index = 0
		self.episode_start_time = time.time()
		# Reset previous agent position when a new session starts
		self.prev_agent_pos = None

		# Gym reset
		self.observation, info = self.env.reset()
		print("LeRobot session started.")

	def end_session(self, force=False):
		"""
		End the session: if 'force', we assume user wants to abort (success=False).
		Otherwise we consider normal end (like environment done).
		"""
		print("LeRobot session ended.")
		# For now, just save the data
		self.save_le_robot_data()
		self.session_active = False

	def step_session(self):
		"""
		Single environment step in either collection or inference mode.
		Logs data in LeRobot format each frame.
		"""
		done = False
		truncated = False

		# Get the action
		if self.mode == "collection":
			# Mouse-based action in [0..512], or adapt as needed
			mouse_pos = pygame.mouse.get_pos()
			action = np.array([
				mouse_pos[0] * ACTION_LIM / SCREEN_WIDTH,
				mouse_pos[1]* ACTION_LIM / SCREEN_HEIGHT],
				dtype=np.float32)
		else:
			# Inference mode with policy
			agent_pos_np = np.array(self.observation["agent_pos"])
			agent_pos_tensor = torch.from_numpy(agent_pos_np).float()
			if self.prev_agent_pos is None:
				# Duplicate if t-1 does not exist.
				state = torch.cat([agent_pos_tensor, agent_pos_tensor], dim=0).unsqueeze(0)
			else:
				state = torch.cat([self.prev_agent_pos.clone(), agent_pos_tensor], dim=0).unsqueeze(0)
			# Update previous agent position.
			self.prev_agent_pos = agent_pos_tensor.clone()

			# Get the image as a uint8 numpy array, similar to training
			image_array = self.observation["pixels"]
			if not isinstance(image_array, np.ndarray):
				image_array = np.array(image_array, dtype=np.uint8)
			# Convert the array directly to a PIL image.
			image_pil = Image.fromarray(image_array)
			# Apply the same image transform as in training.
			image_tensor = image_transform(image_pil)

			predicted = self.policy_inference.sample_action(state, image_tensor)[0]
			action = predicted.cpu().numpy().astype(np.float32)


		# Step environment
		next_observation, reward, done, truncated, info = self.env.step(action)

		# Convert to python lists
		obs_list = self.observation["agent_pos"].tolist()
		act_list = action.tolist()

		# Log in LeRobot format
		success = bool(done)  # or something more specific
		self.log_le_robot_step(
			observation_state=obs_list,
			action=act_list,
			reward=float(reward),
			done=bool(done),
			success=success
		)

		# Render the environment to the pygame screen
		img_arr = self.env.render()
		if img_arr is not None:
			surface = pygame.surfarray.make_surface(img_arr.swapaxes(0, 1))
			self.screen.blit(surface, (0, 0))

		# Move forward
		self.observation = next_observation

		if done or truncated:
			# Session ended naturally
			self.end_session(force=False)


# -----------------------------------------------------------
# CustomSimulation
# -----------------------------------------------------------
class CustomSimulation(BaseSimulation):
	"""
	Custom pygame-based simulation that can run in either data collection or inference mode,
	logging data in the same LeRobot format (one step per frame).
	"""

	def __init__(self, mode="collection"):
		super().__init__(mode=mode, env_type="custom")

		# Initialize simulation objects
		self.arm = ArmNR(BASE_POS, LINK_LENGTHS, torch.tensor(INITIAL_ANGLES, dtype=torch.float32))
		self.object = Object(random_t_pose())
		# Do not randomize goal. No one else does.
		self.goal_pose = torch.tensor([212, 413, -math.pi / 3.0], dtype=torch.float32)

		# For inference mode: hold generated diffusion action for SEC_PER_SAMPLE sec
		self.policy_inference = None
		self.last_diffusion_actions = None
		self.last_diffusion_update_time = 0.0

		# State for controlling the loop
		self.smoothed_target = None
		self.prev_ee_pos = None
		self.last_target = None
		self.max_target_delta = 10

		self.images_dir = os.path.join(self.training_data_dir, "images")
		os.makedirs(self.images_dir, exist_ok=True)

		if self.mode == "inference":
			recompute_normalization_stats(self.env_type, "custom/")
			self.policy_inference = DiffusionPolicyInference(model_path="custom/diffusion_policy.pth",
													norm_stats_path="custom/normalization_stats.json")
			print("Custom: Inference mode.")
		else:
			print("Custom: Collection mode.")

	def run(self):
		"""
		Main loop:
		 - Press [N] to start a session
		 - Press [Q] to quit
		 - If session is active, we do a step per frame
		"""
		print("Press [N] to start a new custom push session. Press [Q] to quit.")
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
		"""
		Reset sim state, clear logs, mark session as active.
		"""
		self.session_active = True
		self.le_robot_data = []  # empty out previous data
		self.episode_index += 1
		self.frame_index = 0
		self.global_index = 0
		self.episode_start_time = time.time()

		# Reset the object, arm, etc.
		self.object = Object(random_t_pose())
		self.arm.joint_angles = torch.zeros(NUM_JOINTS, dtype=torch.float32)
		self.object.velocity = torch.zeros(2, dtype=torch.float32)
		self.object.angular_velocity = 0.0
		self.smoothed_target = None
		self.prev_ee_pos = None
		self.last_diffusion_actions = None
		self.last_diffusion_update_time = 0.0
		self.last_target = None

		# Temp images path if storing images
		self.temp_images_dir = os.path.join(self.training_data_dir, f"temp_images_{int(time.time())}")
		os.makedirs(self.temp_images_dir, exist_ok=True)

		print("Custom session started.")

	def end_session(self, force=False):
		"""
		Session ends: if forced, we do not consider it success.
		Otherwise, we can consider success if at threshold.
		In all cases, save data.
		"""
		print("Custom session ended.")
		self.save_le_robot_data()  # from BaseSimulation
		self.clean_up_temp_images()
		self.session_active = False

	def step_session(self):
		"""
		One "frame" of the custom environment,
		logging a single step in LeRobot format.
		"""
		dt = 1.0 / FPS

		# Get action from either mouse or inference policy
		target = self.get_target_input()
		# If out of workspace => done but no success
		if torch.norm(target - BASE_POS) > ARM_LENGTH:
			print("Input outside arm workspace => terminating session.")
			# log the final step with success=False
			self.log_le_robot_step(
				observation_state=[0, 0],
				action=target.cpu().numpy().tolist(),
				reward=0.0,
				done=True,
				success=False
			)
			self.end_session(force=True)
			return

		self.smoothed_target = self.update_smoothed_target(target)
		ik_error = self.arm.solve_ik(self.smoothed_target)
		if not torch.isfinite(torch.tensor(ik_error)) or ik_error > 1e4:
			print("Invalid IK => terminating session.")
			self.log_le_robot_step(
				observation_state=[0, 0],
				action=target.cpu().numpy().tolist(),
				reward=0.0,
				done=True,
				success=False
			)
			self.end_session(force=True)
			return

		# Forward kinematics => EE pos
		ee_pos = self.arm.forward_kinematics()
		ee_velocity = compute_ee_velocity(ee_pos, self.prev_ee_pos, dt)
		self.prev_ee_pos = ee_pos.clone()

		# Physics
		self.update_object_physics(ee_pos, ee_velocity)

		# Draw
		self.draw_goal_T()
		self.arm.draw(self.screen)
		self.object.draw(self.screen)
		pygame.draw.circle(self.screen, (0, 200, 0),
						   (int(target[0].item()), int(target[1].item())), 6)

		# Calculate a "reward" as pose error, check success => done if threshold
		pos_error = torch.norm(self.object.pose[:2] - self.goal_pose[:2]).item()
		orient_error = abs(angle_diff(self.object.pose[2], self.goal_pose[2]))
		done = bool(pos_error < GOAL_POS_TOL and orient_error < GOAL_ORIENT_TOL)
		reward = float(pos_error)
		success = bool(done)

		# Log step in LeRobot format
		observation_state = ee_pos.cpu().numpy().tolist()  # or anything else
		action_list = self.last_target.cpu().numpy().tolist() if self.last_target is not None else [0, 0]
		self.log_le_robot_step(
			observation_state=observation_state,
			action=action_list,
			reward=reward,
			done=done,
			success=success
		)

		# If done => end session
		if done:
			print("Goal reached => success = True => ending session.")
			self.end_session(force=False)
		else:
			# Not done => continue
			pass

	# -------------- Helper methods -------------- #

	def clean_up_temp_images(self):
		"""
		Move or delete the images in temp_images_dir as needed.
		"""
		if not hasattr(self, "temp_images_dir") or not os.path.exists(self.temp_images_dir):
			return
		# Here we can do something like move them to self.images_dir if desired
		for file_name in os.listdir(self.temp_images_dir):
			src = os.path.join(self.temp_images_dir, file_name)
			dst = os.path.join(self.images_dir, file_name)
			shutil.move(src, dst)
		os.rmdir(self.temp_images_dir)

	def get_target_input(self) -> torch.Tensor:
		"""
		If inference mode => sample from diffusion policy.
		If collection => mouse position => [0..512].
		"""
		if self.mode == "inference":
			current_time = time.time()
			if (self.last_diffusion_actions is None or
				len(self.last_diffusion_actions) == 0 or
				(current_time - self.last_diffusion_update_time) >= SEC_PER_SAMPLE):

				current_ee = self.arm.forward_kinematics()
				if self.prev_ee_pos is None:
					state = torch.cat([current_ee, current_ee], dim=0).unsqueeze(0)
				else:
					state = torch.cat([self.prev_ee_pos, current_ee], dim=0).unsqueeze(0)

				# some image if needed
				image = self.get_current_image_tensor(current_ee)
				self.last_diffusion_actions = self.policy_inference.sample_action(state, image)
				self.last_diffusion_update_time = current_time

			target = self.last_diffusion_actions[0]
			self.last_diffusion_actions = []
			self.last_target = target
			return target
		else:
			# mouse-based
			mouse_pos = pygame.mouse.get_pos()  # e.g. [0..512]
			action = np.array([
				mouse_pos[0],
				mouse_pos[1]],
				dtype=np.float32)
			target = torch.tensor(action, dtype=torch.float32)
			self.last_target = target
			return target

	def update_smoothed_target(self, target: torch.Tensor) -> torch.Tensor:
		"""
		Smooth the target to avoid abrupt changes.
		"""
		if self.smoothed_target is None:
			return target.clone()
		dist, _ = self.object.compute_contact(target)
		if dist < EE_RADIUS:
			delta = target - self.smoothed_target
			delta_clamped = torch.clamp(delta, -self.max_target_delta, self.max_target_delta)
			return self.smoothed_target + delta_clamped
		return target.clone()

	def draw_goal_T(self):
		"""
		Draw the goal shape in the scene.
		"""
		world_vertices = Object.get_transformed_polygon(self.goal_pose)
		pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
		pygame.draw.polygon(self.screen, __import__("config").GOAL_T_COLOR, pts, width=3)

	def update_object_physics(self, ee_pos, ee_velocity):
		"""
		Check for contact forces, push the object.
		"""
		force, push_direction, dist, contact_pt = self.object.get_contact_force(
			ee_pos, ee_velocity, EE_RADIUS, K_CONTACT, K_DAMPING
		)
		if force is not None:
			torque, true_centroid = self.object.update(
				force, 1.0 / FPS, contact_pt,
				velocity_force=K_DAMPING * (ee_velocity - self.object.velocity)
			)
			self.draw_contact_forces(force, contact_pt, true_centroid, torque)
		else:
			self.object.velocity = torch.zeros(2, dtype=torch.float32)
			self.object.angular_velocity = 0.0

	def draw_contact_forces(self, force, contact_pt, true_centroid, torque):
		"""
		Visual debugging for contact point, forces, etc.
		"""
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
		cp_pt = (contact_x, contact_y)
		pygame.draw.line(self.screen, (255, 0, 0), centroid_pt, cp_pt, 3)
		intermediate_pt = (cp_pt[0], centroid_pt[1])
		pygame.draw.line(self.screen, (0, 0, 255), centroid_pt, intermediate_pt, 2)
		pygame.draw.line(self.screen, (0, 0, 255), intermediate_pt, cp_pt, 2)

		force_scale = 0.005
		force_x_end = (int(cp_pt[0] + force[0].item() * force_scale), cp_pt[1])
		draw_arrow(self.screen, (255, 0, 255), cp_pt, force_x_end,
				   width=3, head_length=8, head_angle=30)
		force_y_end = (cp_pt[0], int(cp_pt[1] + force[1].item() * force_scale))
		draw_arrow(self.screen, (0, 255, 0), cp_pt, force_y_end,
				   width=3, head_length=8, head_angle=30)

		fx_text = self.font.render(f"F_x: {force[0]:.1f}", True, (255, 0, 255))
		fy_text = self.font.render(f"F_y: {force[1]:.1f}", True, (0, 255, 0))
		self.screen.blit(fx_text, (cp_pt[0] + 10, cp_pt[1] - 20))
		self.screen.blit(fy_text, (cp_pt[0] + 10, cp_pt[1] + 5))

		torque_text = self.font.render(f"Torque: {torque:.2f}", True, (0, 0, 0))
		self.screen.blit(torque_text, (cp_pt[0] + 10, cp_pt[1] + 30))

	def get_current_image_tensor(self, target=None, for_save=False):
		"""
		If needed, capture a processed simulation image.
		If 'for_save' is True, return a PIL image. Otherwise, a model-ready tensor.
		"""
		if target is not None:
			surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
			surface.fill(BACKGROUND_COLOR)
			world_vertices = Object.get_transformed_polygon(self.goal_pose)
			pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
			pygame.draw.polygon(surface, __import__("config").GOAL_T_COLOR, pts, width=3)
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

		if for_save:
			return pil_img
		else:
			return image_transform(pil_img).unsqueeze(0)


# ----------------------------------------------------------------------
# Factory function
# ----------------------------------------------------------------------
def create_simulation(mode: str, env_type: str) -> BaseSimulation:
	"""
	Construct the correct simulation based on env_type.
	"""
	if env_type == "lerobot":
		return LeRobotSimulation(mode=mode)
	else:
		return CustomSimulation(mode=mode)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run Push Simulation in data collection or inference mode.")
	parser.add_argument("--mode", type=str, default="collection",
						choices=["collection", "inference"],
						help="Choose 'collection' or 'inference' mode.")
	parser.add_argument("--env", type=str, default="custom",
						choices=["custom", "lerobot"],
						help="Select environment type: 'custom' or 'lerobot'.")
	args = parser.parse_args()

	sim = create_simulation(args.mode, args.env)
	sim.run()

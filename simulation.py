#!/usr/bin/env python

import argparse
import pygame
import json
import os
import time
import torch
from config import *
from utils import random_t_pose, angle_diff, draw_arrow
from arm import ArmNR
from object import Object
from policy_inference import DiffusionPolicyInference
from einops import rearrange
from PIL import Image  # new import for image processing
import torchvision.transforms as transforms  # new import for image transforms

# Define transform for images (similar to training)
image_transform = transforms.Compose([
	transforms.Resize((IMG_RES, IMG_RES)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Simulation:
	"""
	Combined simulation that can run in either data collection mode or inference mode.
	
	In data collection mode ("collection"):
	  - The simulation uses mouse (or external input) for target input.
	  - The demo data is logged with the manual target.
	
	In inference mode ("inference"):
	  - The simulation loads a trained diffusion policy.
	  - It uses the diffusion policy to generate an EE action based on the desired pose.
	  - The generated target (diffusion_action) is logged.
	"""
	def __init__(self, mode="collection"):
		self.mode = mode  # Mode can be "collection" or "inference"
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		pygame.display.set_caption("Push Simulation")
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("Arial", 20)
		
		# Initialize simulation objects.
		self.arm = ArmNR(BASE_POS, LINK_LENGTHS, torch.tensor(INITIAL_ANGLES, dtype=torch.float32))
		self.object = Object(random_t_pose())
		self.goal_pose = random_t_pose()
		
		self.session_active = False
		self.demo_data = []  # Raw per-frame logs (downsampled to 10Hz)
		self.session_start_time = None
		self.smoothed_target = None
		self.prev_ee_pos = None
		
		# For inference mode: hold generated diffusion action for SEC_PER_SAMPLE sec.
		self.last_diffusion_actions = None
		self.last_diffusion_update_time = 0.0
		
		# Flag to indicate when realistic data logging starts (after arm initialization).
		self.session_initialized = False  # new flag for skipping initial unrealistic data
		
		# Create a directory for images.
		self.images_dir = os.path.join(TRAINING_DATA_DIR, "images")
		os.makedirs(self.images_dir, exist_ok=True)
		
		# Simulation parameter: maximum change in target per frame.
		self.max_target_delta = 10.0
		
		# In inference mode, load the trained diffusion policy.
		if self.mode == "inference":
			self.policy_inference = DiffusionPolicyInference(model_path="diffusion_policy.pth")
			print("Running in Inference Mode")
		else:
			print("Running in Data Collection Mode")

	def update_smoothed_target(self, target: torch.Tensor) -> torch.Tensor:
		"""
		Smooth the target input to avoid abrupt changes.
		"""
		if self.smoothed_target is None:
			return target.clone()
		dist, _ = self.object.compute_contact(target)
		if dist < EE_RADIUS:
			delta = target - self.smoothed_target
			delta_clamped = torch.clamp(delta, -self.max_target_delta, self.max_target_delta)
			return self.smoothed_target + delta_clamped
		return target.clone()

	@staticmethod
	def compute_ee_velocity(ee_pos: torch.Tensor, prev_ee_pos: torch.Tensor, dt: float) -> torch.Tensor:
		"""
		Compute the EE velocity based on the current and previous positions.
		"""
		if prev_ee_pos is None:
			return torch.zeros_like(ee_pos)
		return (ee_pos - prev_ee_pos) / dt

	def start_session(self):
		"""
		Initialize a new push session.
		"""
		self.session_active = True
		self.demo_data = []  # Reset raw log
		self.session_start_time = time.time()
		self.object = Object(random_t_pose())
		self.goal_pose = random_t_pose()
		self.arm.joint_angles = torch.zeros(NUM_JOINTS, dtype=torch.float32)
		self.object.velocity = torch.zeros(2, dtype=torch.float32)
		self.object.angular_velocity = 0.0
		self.smoothed_target = None
		self.prev_ee_pos = None
		self.last_diffusion_actions = None
		self.last_diffusion_update_time = 0.0
		self.session_initialized = False  # reset initialization flag
		print("New push session started.")

	def get_current_image_tensor(self):
		"""
		Capture the current screen, convert it to a PIL image, and apply image transforms.
		Returns a tensor of shape (1, 3, IMG_RES, IMG_RES).
		"""
		data = pygame.image.tostring(self.screen, 'RGB')
		img = Image.frombytes('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), data)
		image_tensor = image_transform(img)
		return image_tensor.unsqueeze(0)  # add batch dimension

	def get_target_input(self):
		"""
		Get the target position for the current simulation tick.

		In inference mode:
		- If the buffer of predicted actions is empty or if it's time to resample,
			build the state (from the last two end-effector positions) and capture the current image.
		- Call the diffusion policy to obtain a full predicted sequence of actions.
		- Store this sequence in a buffer and then pop the first action from the buffer to use as the target.
		
		In collection mode:
		- Use the mouse position.
		"""
		if self.mode == "inference":
			current_time = time.time()
			# If no actions are buffered or it's time to generate a new sequence, sample a new prediction.
			if (self.last_diffusion_actions is None or len(self.last_diffusion_actions) == 0 or
				(current_time - self.last_diffusion_update_time) >= SEC_PER_SAMPLE):
				
				# Build state condition from EE positions (using t-1 and t)
				current_ee = self.arm.forward_kinematics()
				if self.prev_ee_pos is None:
					state = torch.cat([current_ee, current_ee], dim=0).unsqueeze(0)  # shape (1,4)
				else:
					state = torch.cat([self.prev_ee_pos, current_ee], dim=0).unsqueeze(0)
				
				# Capture the current image from the simulation
				image = self.get_current_image_tensor()  # shape (1, 3, IMG_RES, IMG_RES)
				
				# Generate the full predicted sequence of actions and store in the buffer
				self.last_diffusion_actions = self.policy_inference.sample_action(state, image)
				self.last_diffusion_update_time = current_time
			
			# Pop the first action from the buffer to execute this tick
			target = self.last_diffusion_actions[0]
			# Remove the executed action from the buffer
			self.last_diffusion_actions = self.last_diffusion_actions[1:]
			return target
		else:
			# In collection mode, use mouse input.
			return torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32)

	def draw_goal_T(self):
		"""
		Draw the goal shape using the transformation provided by Object.
		"""
		world_vertices = Object.get_transformed_polygon(self.goal_pose)
		pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
		pygame.draw.polygon(self.screen, __import__("config").GOAL_T_COLOR, pts, width=3)

	def run(self):
		"""
		Run the main simulation loop.
		
		Press [N] to start a new session.
		The session terminates if the input (or generated target) is outside the arm workspace,
		if the inverse kinematics solution is invalid, or when the object reaches the goal.
		"""
		print("Press [N] to start a new push session.")
		print("Session saves if goal tolerances are met; session quits if IK fails or if input leaves workspace.")
		os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					return
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_n and not self.session_active:
						self.start_session()
			
			self.screen.fill(BACKGROUND_COLOR)
			self.draw_goal_T()
			
			if self.session_active:
				target = self.get_target_input()
				if torch.norm(target - BASE_POS) > ARM_LENGTH:
					print("Input outside arm workspace. Terminating session.")
					self.session_active = False
					self.demo_data = []
					continue
				
				self.smoothed_target = self.update_smoothed_target(target)
				ik_error = self.arm.solve_ik(self.smoothed_target)
				if not torch.isfinite(torch.tensor(ik_error)) or ik_error > 1e4:
					print("Invalid IK result detected. Terminating session.")
					self.session_active = False
					self.demo_data = []
					continue
				
				ee_pos = self.arm.forward_kinematics()
				dt = 1.0 / FPS
				ee_velocity = Simulation.compute_ee_velocity(ee_pos, self.prev_ee_pos, dt)
				self.prev_ee_pos = ee_pos.clone()
				
				force, push_direction, dist, contact_pt = self.object.get_contact_force(
					ee_pos, ee_velocity, EE_RADIUS, K_CONTACT, K_DAMPING)
				
				if force is not None:
					torque, true_centroid = self.object.update(force, dt, contact_pt, velocity_force=K_DAMPING * (ee_velocity - self.object.velocity))
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
					draw_arrow(self.screen, (255, 0, 255), cp_pt, force_x_end, width=3, head_length=8, head_angle=30)
					force_y_end = (cp_pt[0], int(cp_pt[1] + force[1].item() * force_scale))
					draw_arrow(self.screen, (0, 255, 0), cp_pt, force_y_end, width=3, head_length=8, head_angle=30)
					fx_text = self.font.render(f"F_x: {force[0]:.1f}", True, (255, 0, 255))
					fy_text = self.font.render(f"F_y: {force[1]:.1f}", True, (0, 255, 0))
					self.screen.blit(fx_text, (cp_pt[0] + 10, cp_pt[1] - 20))
					self.screen.blit(fy_text, (cp_pt[0] + 10, cp_pt[1] + 5))
					torque_text = self.font.render(f"Torque: {torque:.2f}", True, (0, 0, 0))
					self.screen.blit(torque_text, (cp_pt[0] + 10, cp_pt[1] + 30))
				else:
					self.object.velocity = torch.zeros(2, dtype=torch.float32)
					self.object.angular_velocity = 0.0
				
				self.arm.draw(self.screen)
				self.object.draw(self.screen)
				pygame.draw.circle(self.screen, (0, 200, 0),
								   (int(target[0].item()), int(target[1].item())), 6)
				capture_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
				capture_surface.fill(BACKGROUND_COLOR)
				world_vertices = Object.get_transformed_polygon(self.goal_pose)
				pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
				pygame.draw.polygon(capture_surface, __import__("config").GOAL_T_COLOR, pts, width=3)
				self.object.draw(capture_surface)
				ee = self.arm.forward_kinematics()
				pygame.draw.circle(capture_surface, (0, 0, 0),
								   (int(ee[0].item()), int(ee[1].item())), EE_RADIUS)
				pygame.draw.circle(capture_surface, (0, 200, 0),
								   (int(target[0].item()), int(target[1].item())), 6)
				timestamp = time.time() - self.session_start_time
				
				# Ensure unique image filenames to avoid duplicates
				base_filename = f"frame_{timestamp:.3f}"
				img_filename = base_filename + ".png"
				counter = 1
				full_img_path = os.path.join(self.images_dir, img_filename)
				while os.path.exists(full_img_path):
					img_filename = f"{base_filename}_{counter}.png"
					full_img_path = os.path.join(self.images_dir, img_filename)
					counter += 1
				capture_image = pygame.transform.scale(capture_surface, (IMG_RES, IMG_RES))
				pygame.image.save(capture_image, full_img_path)
				
				log_entry = {
					"time": timestamp,
					"goal_pose": self.goal_pose.tolist(),
					"T_pose": self.object.pose.tolist(),
					"EE_pos": ee.tolist(),
					"image": img_filename  # Save image filename
				}
				if self.mode == "inference":
					log_entry["diffusion_action"] = target.tolist()
				else:
					log_entry["action"] = target.tolist()
				# Also log the IK error for debugging purposes.
				log_entry["ik_error"] = ik_error
				
				# Skip logging unrealistic data:
				# Do not log frames from arm initialization until the EE snaps to the target (mouse position)
				# Threshold set to 10.0 for EE proximity to target.
				if self.mode != "inference":
					if not self.session_initialized:
						if torch.norm(ee - target) > 10.0:
							# Skip logging this frame because EE has not yet snapped to the target.
							#print("Skipping logging: waiting for EE to snap to target.")
							pass  # Do not append log entry
						else:
							self.session_initialized = True
							print(f'Session initialized. Logging started at {timestamp:.3f} seconds.')
							if not self.demo_data or (timestamp - self.demo_data[-1]["time"]) >= SEC_PER_SAMPLE:
								self.demo_data.append(log_entry)
					else:
						if not self.demo_data or (timestamp - self.demo_data[-1]["time"]) >= SEC_PER_SAMPLE:
							self.demo_data.append(log_entry)
				
				pos_error = torch.norm(self.object.pose[:2] - self.goal_pose[:2]).item()
				orient_error = abs(angle_diff(self.object.pose[2], self.goal_pose[2]))
				if pos_error < GOAL_POS_TOL and orient_error < GOAL_ORIENT_TOL:
					print("T object reached desired pose. Session complete.")
					training_examples = []
					if len(self.demo_data) >= DEMO_DATA_FRAMES:
						for i in range(1, len(self.demo_data) - WINDOW_SIZE):
							obs = {
								"image": [
									self.demo_data[i - 1]["image"],
									self.demo_data[i]["image"]
								],
								"state": [
									self.demo_data[i - 1]["EE_pos"],
									self.demo_data[i]["EE_pos"]
								]
							}
							actions = []
							for j in range(i - 1, i + WINDOW_SIZE+1):
								key = "diffusion_action" if self.mode == "inference" else "action"
								actions.append(self.demo_data[j][key])
							training_examples.append({
								"observation": obs,
								"action": actions
							})
					# Ensure unique training data filenames to avoid duplicates
					base_filename = f"session_{int(time.time())}_training"
					filename = os.path.join(TRAINING_DATA_DIR, base_filename + ".json")
					counter = 1
					while os.path.exists(filename):
						filename = os.path.join(TRAINING_DATA_DIR, f"{base_filename}_{counter}.json")
						counter += 1
					with open(filename, "w") as f:
						json.dump(training_examples, f, indent=2)
					print(f"Training session data saved to {filename}.")
					self.session_active = False
			
			else:
				text = self.font.render("Press [N] to start a new push session", True, (0, 0, 0))
				self.screen.blit(text, (20, 20))
				self.arm.draw(self.screen)
				self.object.draw(self.screen)
			
			pygame.display.flip()
			self.clock.tick(FPS)

if __name__ == "__main__":
	import sys
	parser = argparse.ArgumentParser(description="Run Push Simulation in data collection or inference mode.")
	parser.add_argument("--mode", type=str, default="collection", choices=["collection", "inference"],
						help="Choose 'collection' for data collection mode or 'inference' for diffusion inference mode.")
	args = parser.parse_args()
	
	sim = Simulation(mode=args.mode)
	sim.run()

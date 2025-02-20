#!/usr/bin/env python

import pygame
import json
import os
import time
import torch
from config import (SCREEN_WIDTH, SCREEN_HEIGHT, BACKGROUND_COLOR, FPS,
					GOAL_POS_TOL, GOAL_ORIENT_TOL, EE_RADIUS, ARM_LENGTH, BASE_POS,
					K_CONTACT, K_DAMPING, NUM_JOINTS, LINK_LENGTHS)
from utils import random_t_pose, angle_diff, draw_arrow
from arm import ArmNR
from t_object import TObject

class Simulation:
	"""
	Encapsulates the simulation. Handles input (from mouse or an external agent),
	runs the update loop, and draws the arm and object.
	"""
	def __init__(self, input_provider=None):
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		pygame.display.set_caption("Push T Data Collection")
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("Arial", 20)
		self.input_provider = input_provider  # Callable returning a torch.Tensor

		# Initialize simulation objects.
		self.arm = ArmNR(BASE_POS, LINK_LENGTHS)
		self.T_object = TObject(random_t_pose())
		self.goal_pose = random_t_pose()

		self.session_active = False
		self.demo_data = []
		self.session_start_time = None
		self.smoothed_target = None
		self.prev_ee_pos = None

		# Simulation parameters
		self.max_target_delta = 10.0  # Maximum change in target per frame

	def update_smoothed_target(self, target: torch.Tensor) -> torch.Tensor:
		"""
		Smooth the target input so that abrupt changes (especially when near the object)
		are clamped. If the target is in contact with the T object, only allow a small change.
		"""
		if self.smoothed_target is None:
			return target.clone()
		# Check contact distance between the target and the object's polygon.
		dist, _ = self.T_object.compute_contact(target)
		if dist < EE_RADIUS:
			delta = target - self.smoothed_target
			delta_clamped = torch.clamp(delta, -self.max_target_delta, self.max_target_delta)
			return self.smoothed_target + delta_clamped
		return target.clone()

	@staticmethod
	def compute_ee_velocity(ee_pos: torch.Tensor, prev_ee_pos: torch.Tensor, dt: float) -> torch.Tensor:
		"""
		Compute the end-effector velocity based on the current and previous positions.
		"""
		if prev_ee_pos is None:
			return torch.zeros_like(ee_pos)
		return (ee_pos - prev_ee_pos) / dt

	def start_session(self):
		"""
		Initialize a new push session.
		"""
		self.session_active = True
		self.demo_data = []
		self.session_start_time = time.time()
		self.T_object = TObject(random_t_pose())
		self.goal_pose = random_t_pose()
		self.arm.joint_angles = torch.zeros(NUM_JOINTS, dtype=torch.float32)
		self.T_object.velocity = torch.zeros(2, dtype=torch.float32)
		self.T_object.angular_velocity = 0.0
		self.smoothed_target = None
		self.prev_ee_pos = None
		print("New push session started.")

	def get_target_input(self):
		"""
		Get the target position either from an external input provider or from the mouse.
		"""
		if self.input_provider is not None:
			return self.input_provider()
		else:
			return torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32)

	def draw_goal_T(self):
		"""
		Draw the goal T shape using the static transformation provided by TObject.
		"""
		from t_object import TObject
		world_vertices = TObject.get_transformed_polygon(self.goal_pose)
		pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
		pygame.draw.polygon(self.screen, 
							__import__("config").GOAL_T_COLOR, pts, width=3)

	def run(self):
		print("Press [N] to start a new push session.")
		print("Session saves if goal tolerances are met; session quits if IK fails or if input leaves workspace.")
		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					if self.demo_data:
						os.makedirs("training_data", exist_ok=True)  # Create the directory if it doesn't exist
						filename = os.path.join("training_data", f"session_{int(time.time())}.json")
						with open(filename, "w") as f:
							json.dump(self.demo_data, f, indent=2)
						print(f"Session data saved to {filename}.")
					pygame.quit()
					return
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_n and not self.session_active:
						self.start_session()

			self.screen.fill(BACKGROUND_COLOR)
			self.draw_goal_T()

			if self.session_active:
				target = self.get_target_input()
				# Terminate session if input is outside arm workspace.
				if torch.norm(target - BASE_POS) > ARM_LENGTH:
					print("Input outside arm workspace. Terminating session.")
					self.session_active = False
					self.demo_data = []
					continue

				# Update the smoothed target to avoid abrupt jumps.
				self.smoothed_target = self.update_smoothed_target(target)

				# Solve inverse kinematics to drive the arm toward the smoothed target.
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

				# Compute the contact force using the T object's helper.
				force, push_direction, dist, contact_pt = self.T_object.get_contact_force(
					ee_pos, ee_velocity, EE_RADIUS, K_CONTACT, K_DAMPING)

				if force is not None:
					# Update the object's state using the computed force.
					torque, true_centroid = self.T_object.update(force, dt, contact_pt, velocity_force=K_DAMPING * (ee_velocity - self.T_object.velocity))

					# Draw contact markers and force vectors.
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

					# Draw intermediate lines for visualization.
					intermediate_pt = (cp_pt[0], centroid_pt[1])
					pygame.draw.line(self.screen, (0, 0, 255), centroid_pt, intermediate_pt, 2)
					pygame.draw.line(self.screen, (0, 0, 255), intermediate_pt, cp_pt, 2)

					# Scale and draw force vectors.
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
					self.T_object.velocity = torch.zeros(2, dtype=torch.float32)
					self.T_object.angular_velocity = 0.0

				# Draw the arm and object.
				self.arm.draw(self.screen)
				self.T_object.draw(self.screen)
				pygame.draw.circle(self.screen, (0, 200, 0),
								   (int(target[0].item()), int(target[1].item())), 6)

				# Log simulation data.
				timestamp = time.time() - self.session_start_time
				self.demo_data.append({
					"time": timestamp,
					"input_target": target.tolist(),
					"joint_angles": self.arm.joint_angles.tolist(),
					"ee_pos": ee_pos.tolist(),
					"T_pose": self.T_object.pose.tolist(),
					"ik_error": ik_error
				})

				pos_error = torch.norm(self.T_object.pose[:2] - self.goal_pose[:2]).item()
				orient_error = abs(angle_diff(self.T_object.pose[2], self.goal_pose[2]))
				if pos_error < GOAL_POS_TOL and orient_error < GOAL_ORIENT_TOL:
					print("T object reached desired pose. Session complete.")
					os.makedirs("training_data", exist_ok=True)  # Create the directory if it doesn't exist
					filename = os.path.join("training_data", f"session_{int(time.time())}.json")
					with open(filename, "w") as f:
						json.dump(self.demo_data, f, indent=2)
					print(f"Session data saved to {filename}.")
					self.session_active = False

			else:
				text = self.font.render("Press [N] to start a new push session", True, (0, 0, 0))
				self.screen.blit(text, (20, 20))
				self.arm.draw(self.screen)
				self.T_object.draw(self.screen)

			pygame.display.flip()
			self.clock.tick(FPS)

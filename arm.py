#!/usr/bin/env python

import math
import torch
import numpy as np
import pygame
from scipy.optimize import least_squares
from config import EE_RADIUS, LINK_COLLISION_THRESHOLD, LINK_COLLISION_WEIGHT, ARM_COLOR

class ArmNR:
	"""
	A multi-link arm that computes forward kinematics and solves inverse kinematics.
	Uses SciPy's least_squares optimizer to minimize a cost that includes the end-effector
	position error and penalties for potential self-collisions.
	"""
	def __init__(self, base_pos: torch.Tensor, link_lengths: list, initial_angles=None):
		self.base_pos = base_pos
		self.link_lengths = link_lengths
		self.num_joints = len(link_lengths)
		if initial_angles is None:
			self.joint_angles = torch.zeros(self.num_joints, dtype=torch.float32)
		else:
			self.joint_angles = initial_angles.clone()

	def compute_joint_positions(self) -> list:
		"""
		Compute (x, y) positions for each joint (including the end-effector)
		using simple forward kinematics.
		"""
		positions = [self.base_pos.clone()]
		total_angle = 0.0
		current_pos = self.base_pos.clone()
		for i in range(self.num_joints):
			total_angle += self.joint_angles[i]
			dx = self.link_lengths[i] * torch.cos(total_angle)
			dy = self.link_lengths[i] * torch.sin(total_angle)
			current_pos = current_pos + torch.tensor([dx, dy])
			positions.append(current_pos.clone())
		return positions

	def compute_joint_positions_np(self, angles: np.ndarray) -> list:
		"""
		Same as compute_joint_positions but operating on a NumPy array of angles.
		Used during the least-squares optimization.
		"""
		positions = []
		current_pos = np.array(self.base_pos.numpy())
		positions.append(current_pos.copy())
		total_angle = 0.0
		for i in range(self.num_joints):
			total_angle += angles[i]
			dx = self.link_lengths[i] * math.cos(total_angle)
			dy = self.link_lengths[i] * math.sin(total_angle)
			current_pos = current_pos + np.array([dx, dy])
			positions.append(current_pos.copy())
		return positions

	def forward_kinematics(self) -> torch.Tensor:
		"""
		Return the (x, y) position of the end-effector.
		"""
		return self.compute_joint_positions()[-1]

	def solve_ik(self, target: torch.Tensor) -> float:
		"""
		Solve inverse kinematics to move the end-effector towards target.
		The cost function includes the distance error as well as collision penalties.
		"""
		target_np = target.numpy()

		def point_to_segment_distance(P, A, B):
			# Compute the distance from point P to the segment AB.
			AP = P - A
			AB = B - A
			ab_sq = np.dot(AB, AB)
			if ab_sq == 0:
				return np.linalg.norm(P - A)
			t = np.dot(AP, AB) / ab_sq
			t = max(0.0, min(1.0, t))
			projection = A + t * AB
			return np.linalg.norm(P - projection)

		def segment_distance(A, B, C, D):
			# Returns the minimal distance between segments AB and CD.
			d1 = point_to_segment_distance(A, C, D)
			d2 = point_to_segment_distance(B, C, D)
			d3 = point_to_segment_distance(C, A, B)
			d4 = point_to_segment_distance(D, A, B)
			return min(d1, d2, d3, d4)

		def residuals(x):
			positions = self.compute_joint_positions_np(x)
			ee_pos = positions[-1]
			res = list(ee_pos - target_np)
			num_links = len(positions) - 1
			# Penalty for self-collision: For non-adjacent links.
			for i in range(num_links):
				for j in range(i + 2, num_links):
					A, B = positions[i], positions[i+1]
					C, D = positions[j], positions[j+1]
					d = segment_distance(A, B, C, D)
					if d < LINK_COLLISION_THRESHOLD:
						res.append(LINK_COLLISION_WEIGHT * (LINK_COLLISION_THRESHOLD - d))
					else:
						res.append(0.0)
			return np.array(res)

		x0 = self.joint_angles.numpy()
		result = least_squares(residuals, x0)
		self.joint_angles = torch.tensor(result.x, dtype=torch.float32)
		return result.cost

	def draw(self, surface, color=None, joint_radius=None, width=5):
		"""
		Draw the arm on the given pygame surface.
		"""
		if color is None:
			color = ARM_COLOR
		if joint_radius is None:
			joint_radius = int(EE_RADIUS)
		positions = self.compute_joint_positions()
		pts = [(int(pos[0].item()), int(pos[1].item())) for pos in positions]
		for i in range(len(pts) - 1):
			pygame.draw.line(surface, color, pts[i], pts[i+1], width)
		pygame.draw.circle(surface, (0, 0, 0), pts[-1], joint_radius, 2)
		return positions[-1]

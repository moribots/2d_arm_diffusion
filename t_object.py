#!/usr/bin/env python

import torch
import pygame
from einops import rearrange
from config import M_T, T_COLOR, LINEAR_DAMPING, ANGULAR_DAMPING

def polygon_moi(vertices: torch.Tensor, mass: float) -> float:
	"""
	Compute the moment of inertia (MOI) for a polygon using the formula:
	  I = density * (numerator) / 12
	where density is mass divided by the polygon area.
	"""
	verts = vertices.numpy()
	n = verts.shape[0]
	A = 0.0
	numerator = 0.0
	for i in range(n):
		x_i, y_i = verts[i]
		x_next, y_next = verts[(i+1) % n]
		cross = x_i * y_next - x_next * y_i
		A += cross
		numerator += cross * (x_i**2 + x_i*x_next + x_next**2 + y_i**2 + y_i*y_next + y_next**2)
	A = abs(A) / 2.0
	density = mass / A
	I = density * numerator / 12.0
	return I

class TObject:
	"""
	A T-shaped object that can be pushed by the arm.
	Contains geometry, drawing, and physics update methods.
	"""
	# Define the T shape vertices; then scale and recenter so that the centroid is at the origin.
	T_SCALE = 2.0
	base_vertices = torch.tensor([
		[-40.0, -10.0],
		[40.0, -10.0],
		[40.0, 10.0],
		[10.0, 10.0],
		[10.0, 70.0],
		[-10.0, 70.0],
		[-10.0, 10.0],
		[-40.0, 10.0]
	], dtype=torch.float32)
	local_vertices = base_vertices * T_SCALE
	centroid = torch.mean(local_vertices, dim=0)
	local_vertices_adjusted = local_vertices - centroid

	def __init__(self, pose: torch.Tensor):
		self.pose = pose.clone()  # [x, y, theta]
		self.velocity = torch.zeros(2, dtype=torch.float32)
		self.angular_velocity = 0.0
		# Compute moment of inertia for rotation dynamics.
		self.moi = polygon_moi(TObject.local_vertices_adjusted, M_T)

	def get_polygon(self):
		"""
		Compute the world coordinates of the T object's polygon by rotating the local vertices
		and translating by the object's pose. Uses einops to rearrange for a matrix multiply.
		"""
		theta = self.pose[2]
		cos_t = torch.cos(theta)
		sin_t = torch.sin(theta)
		R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
		# Using einops to rearrange vertices to (2, N) for matrix multiplication.
		vertices = rearrange(TObject.local_vertices_adjusted, "n d -> d n")
		rotated = torch.matmul(R, vertices)  # shape: (2, N)
		rotated = rearrange(rotated, "d n -> n d")
		return rotated + self.pose[:2]

	@staticmethod
	def get_transformed_polygon(pose: torch.Tensor) -> torch.Tensor:
		"""
		Return the transformed polygon (world coordinates) for a given pose.
		Uses einops for matrix manipulation.
		"""
		theta = pose[2]
		cos_t = torch.cos(theta)
		sin_t = torch.sin(theta)
		R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
		vertices = rearrange(TObject.local_vertices_adjusted, "n d -> d n")
		rotated = torch.matmul(R, vertices)
		rotated = rearrange(rotated, "d n -> n d")
		return rotated + pose[:2]

	def compute_centroid(self):
		"""
		Compute the centroid of the object's polygon using the standard formula
		for polygons (weighted by the cross product).
		"""
		poly = self.get_polygon()
		n = poly.shape[0]
		A = 0.0
		Cx = 0.0
		Cy = 0.0
		for i in range(n):
			j = (i + 1) % n
			xi, yi = poly[i][0], poly[i][1]
			xj, yj = poly[j][0], poly[j][1]
			cross = xi * yj - xj * yi
			A += cross
			Cx += (xi + xj) * cross
			Cy += (yi + yj) * cross
		A = A / 2.0
		Cx = Cx / (6.0 * A)
		Cy = Cy / (6.0 * A)
		return torch.tensor([Cx, Cy], dtype=torch.float32)

	def update(self, force: torch.Tensor, dt: float, contact_pt: torch.Tensor = None, velocity_force: torch.Tensor = None):
		"""
		Update the object's state using Euler integration.
		Linear dynamics:
		  a = F/m   →   v += a * dt, then apply damping, then update position.
		Angular dynamics:
		  torque = r x F, computed about the current centroid.
		  Angular acceleration = torque / moi.
		"""
		# Linear update
		acceleration = force / M_T
		self.velocity = self.velocity + acceleration * dt
		self.velocity = self.velocity * LINEAR_DAMPING
		self.pose[:2] = self.pose[:2] + self.velocity * dt

		# Angular update (using 2D cross product)
		if contact_pt is not None:
			true_centroid = self.compute_centroid()
			r = contact_pt - true_centroid
			# Use velocity_force for torque if available
			force_for_torque = velocity_force if velocity_force is not None else force
			torque = r[0] * force_for_torque[1] - r[1] * force_for_torque[0]
		else:
			torque = 0.0

		angular_acceleration = torque / self.moi
		self.angular_velocity = self.angular_velocity + angular_acceleration * dt
		self.angular_velocity = self.angular_velocity * ANGULAR_DAMPING
		self.pose[2] = self.pose[2] + self.angular_velocity * dt

		return torque, self.compute_centroid()

	def draw(self, surface, joint_radius=8, centroid_radius=4, centroid_color=(0, 0, 0)):
		"""
		Draw the T object and its centroid on the given pygame surface.
		"""
		polygon = self.get_polygon()
		pts = [(int(pos[0].item()), int(pos[1].item())) for pos in polygon]
		pygame.draw.polygon(surface, T_COLOR, pts)
		centroid = self.compute_centroid()
		centroid_pt = (int(centroid[0].item()), int(centroid[1].item()))
		pygame.draw.circle(surface, centroid_color, centroid_pt, centroid_radius)
		return polygon

	def compute_contact(self, ee_pos: torch.Tensor) -> (float, torch.Tensor):
		"""
		Compute the minimum distance between the object's edges and the given point (end-effector).
		Returns the minimum distance and the corresponding closest point on the polygon.
		"""
		polygon = self.get_polygon()
		num_vertices = polygon.shape[0]
		min_dist = float('inf')
		closest_point = None
		for i in range(num_vertices):
			A = polygon[i]
			B = polygon[(i+1) % num_vertices]
			AB = B - A
			AB_norm_sq = torch.dot(AB, AB)
			if AB_norm_sq > 0:
				t = torch.clamp(torch.dot(ee_pos - A, AB) / AB_norm_sq, 0.0, 1.0)
			else:
				t = 0.0
			proj = A + t * AB
			dist = torch.norm(ee_pos - proj)
			if dist < min_dist:
				min_dist = dist.item()
				closest_point = proj
		return min_dist, closest_point

	def get_contact_force(self, ee_pos: torch.Tensor, ee_velocity: torch.Tensor,
						  ee_radius: float, k_contact: float, k_damping: float):
		"""
		Compute the force applied to the object upon contact with the end-effector.
		- If the end-effector is within the EE_RADIUS, a penetration force is computed.
		- A damping force based on the relative velocity is added.
		Returns:
			total_force: The net force vector.
			push_direction: The normalized direction in which the force is applied.
			dist: The computed distance from the end-effector to the polygon.
			contact_pt: The closest point on the polygon.
		"""
		dist, contact_pt = self.compute_contact(ee_pos)
		if dist >= ee_radius:
			return None, None, dist, contact_pt
		raw_push_direction = contact_pt - ee_pos
		# If the computed push direction is nearly zero, fallback to using the vector from object center.
		if torch.norm(raw_push_direction) < 1e-3:
			fallback_dir = contact_pt - self.pose[:2]
			norm_fb = torch.norm(fallback_dir)
			push_direction = fallback_dir / norm_fb if norm_fb > 0 else torch.tensor([0.0, 0.0])
		else:
			push_direction = raw_push_direction / torch.norm(raw_push_direction)
		penetration = ee_radius - dist
		# The penetration force uses a linear spring model: F = k * penetration.
		force_from_penetration = k_contact * penetration * push_direction
		relative_velocity = ee_velocity - self.velocity
		velocity_force = k_damping * relative_velocity
		total_force = force_from_penetration + velocity_force
		return total_force, push_direction, dist, contact_pt

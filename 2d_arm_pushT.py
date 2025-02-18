import pygame
import sys
import json
import time
import math
import random

import torch
import torch.nn.functional as F
from einops import rearrange
import math
import random
import numpy as np
import torch
from scipy.optimize import least_squares

# ------------------------- Global Settings -------------------------
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
BACKGROUND_COLOR = (255, 255, 255)
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)          # Active T color
GOAL_T_COLOR = (0, 200, 0)       # Goal T outline color
FPS = 120

# End-Effector Parameters
EE_RADIUS = 16.0

# Contact Mechanics Parameters
K_CONTACT = 1000.0    # Contact stiffness
M_T = 100.0           # Mass of T block
ANGULAR_DAMPING = 100.0
LINEAR_DAMPING = 0.1
MAX_PENETRATION = 0.5
MAX_T_VEL = 100.0      # Max linear velocity
MAX_T_ANG_VEL = 2.5   # Max angular velocity

# Goal tolerances for ending the session
GOAL_POS_TOL = 2.0
GOAL_ORIENT_TOL = 0.2

# Arm Base Position (offset in the window)
BASE_POS = torch.tensor([300.0, 300.0], dtype=torch.float32)

# ------------------------- Utility Functions -------------------------
def random_t_pose():
	"""
	Generate a random T-object pose (x, y, theta) within screen bounds.
	A margin is used to keep the object fully visible.
	"""
	margin = 100  
	x = random.uniform(margin, SCREEN_WIDTH - margin)
	y = random.uniform(margin, SCREEN_HEIGHT - margin)
	theta = random.uniform(-math.pi, math.pi)
	return torch.tensor([x, y, theta], dtype=torch.float32)

def angle_diff(a: float, b: float) -> float:
	"""
	Compute the difference between two angles (radians), wrapped to [-pi, pi].
	"""
	diff = (a - b + torch.pi) % (2 * torch.pi) - torch.pi
	return diff.item()

def draw_arrow(surface, color, start, end, width=3, head_length=10, head_angle=30):
	"""
	Draw an arrow on the surface from start to end.
	"""
	pygame.draw.line(surface, color, start, end, width)
	dx = end[0] - start[0]
	dy = end[1] - start[1]
	angle = math.atan2(dy, dx)
	left_angle = angle + math.radians(head_angle)
	right_angle = angle - math.radians(head_angle)
	left_end = (end[0] - head_length * math.cos(left_angle),
				end[1] - head_length * math.sin(left_angle))
	right_end = (end[0] - head_length * math.cos(right_angle),
				 end[1] - head_length * math.sin(right_angle))
	pygame.draw.line(surface, color, end, left_end, width)
	pygame.draw.line(surface, color, end, right_end, width)

# ------------------------- ArmNR Class -------------------------
# --- Global Parameters for the 3R Arm and Collision ---
LINK_LENGTHS = [150.0, 150.0, 150.0, 150.0, 150.0]  # Example link lengths; adjust as desired.
NUM_JOINTS = len(LINK_LENGTHS)
ARM_LENGTH = sum(LINK_LENGTHS)

# Parameters for link collision avoidance (if needed)
LINK_COLLISION_THRESHOLD = 20.0  # Minimum allowed distance between non-adjacent links
LINK_COLLISION_WEIGHT = 5.0        # Weight for the collision penalty

class ArmNR:
	"""
	A 3-revolute joint (3R) arm that computes forward kinematics and solves inverse kinematics
	using SciPy's least_squares optimizer. Optionally, it adds a penalty for link collisions.
	"""
	def __init__(self, base_pos: torch.Tensor, link_lengths: list, initial_angles=None):
		self.base_pos = base_pos  # Base position (e.g., in screen coordinates)
		self.link_lengths = link_lengths
		self.num_joints = len(link_lengths)
		if initial_angles is None:
			self.joint_angles = torch.zeros(self.num_joints, dtype=torch.float32)
		else:
			self.joint_angles = initial_angles.clone()

	def compute_joint_positions(self) -> list:
		"""
		Compute the (x, y) positions for each joint (including the end-effector)
		starting from the base. Returns a list of torch.Tensor.
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
		Same as compute_joint_positions but using a given numpy array of joint angles.
		Returns joint positions as numpy arrays (for use in SciPy optimization).
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
		Return the (x, y) position of the end-effector computed from the current joint angles.
		"""
		return self.compute_joint_positions()[-1]

	def solve_ik(self, target: torch.Tensor) -> float:
		"""
		Solve the inverse kinematics problem using SciPy's least_squares.
		The residual function includes:
		  1. The error between the current end-effector position and the target.
		  2. A penalty for any pair of non-adjacent links whose distance falls below a threshold.
		Returns the final cost.
		"""
		target_np = np.array(target.numpy())

		def point_to_segment_distance(P, A, B):
			"""
			Compute the minimum distance from point P to the segment AB.
			"""
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
			"""
			Compute the minimum distance between segments AB and CD.
			"""
			d1 = point_to_segment_distance(A, C, D)
			d2 = point_to_segment_distance(B, C, D)
			d3 = point_to_segment_distance(C, A, B)
			d4 = point_to_segment_distance(D, A, B)
			return min(d1, d2, d3, d4)

		def residuals(x):
			# Compute joint positions using current angles.
			positions = self.compute_joint_positions_np(x)
			# End-effector residual: difference between current and target positions.
			ee_pos = positions[-1]
			res = list(ee_pos - target_np)
			# For a 3R arm there are 3 links: link 0: (positions[0], positions[1]),
			# link 1: (positions[1], positions[2]), link 2: (positions[2], positions[3])
			# We penalize collisions only between non-adjacent links.
			num_links = len(positions) - 1
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

		# Use current joint angles as the initial guess.
		x0 = self.joint_angles.numpy()
		result = least_squares(residuals, x0)
		# Update the arm's joint angles with the solution.
		self.joint_angles = torch.tensor(result.x, dtype=torch.float32)
		return result.cost

	def draw(self, surface, color=(50, 100, 200), joint_radius=int(EE_RADIUS), width=5):
		"""
		Draw the arm onto the given pygame surface. The method computes the joint
		positions and then draws links (as lines) and joints (as circles).
		"""
		positions = self.compute_joint_positions()
		pts = [(int(pos[0].item()), int(pos[1].item())) for pos in positions]
		for i in range(len(pts) - 1):
			pygame.draw.line(surface, color, pts[i], pts[i+1], width)
		for pt in pts:
			pygame.draw.circle(surface, (0, 0, 0), pt, joint_radius, 2)
		return positions[-1]

# ------------------------- TObject Class -------------------------
def polygon_moi(vertices: torch.Tensor, mass: float) -> float:
	"""
	Compute the moment of inertia of a polygon (with vertices in order)
	about its centroid. Assumes uniform density.
	"""
	# Convert vertices to numpy for easier processing.
	verts = vertices.numpy()
	n = verts.shape[0]
	A = 0.0
	numerator = 0.0
	for i in range(n):
		x_i, y_i = verts[i]
		x_next, y_next = verts[(i+1)%n]
		cross = x_i*y_next - x_next*y_i
		A += cross
		numerator += cross * (x_i**2 + x_i*x_next + x_next**2 + y_i**2 + y_i*y_next + y_next**2)
	A = abs(A) / 2.0
	density = mass / A
	I = density * numerator / 12.0
	return I

class TObject:
	"""
	Class representing the T-shaped object.
	"""
	local_vertices = torch.tensor([
		[-40.0, -10.0],
		[40.0, -10.0],
		[40.0, 10.0],
		[10.0, 10.0],
		[10.0, 70.0],
		[-10.0, 70.0],
		[-10.0, 10.0],
		[-40.0, 10.0]
	], dtype=torch.float32)
	T_SCALE = 2.0
	local_vertices = local_vertices * T_SCALE
	# Center the vertices.
	centroid = torch.mean(local_vertices, dim=0)
	local_vertices_adjusted = local_vertices - centroid

	def __init__(self, pose: torch.Tensor):
		self.pose = pose.clone()  # [x, y, theta]
		self.velocity = torch.zeros(2, dtype=torch.float32)
		self.angular_velocity = 0.0
		# Compute the true moment of inertia from the centered polygon.
		self.moi = polygon_moi(TObject.local_vertices_adjusted, M_T)

	def get_polygon(self) -> torch.Tensor:
		theta = self.pose[2]
		cos_t = torch.cos(theta)
		sin_t = torch.sin(theta)
		R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
		rotated = torch.einsum("kj,ij->ki", TObject.local_vertices_adjusted, R)
		world_vertices = rotated + self.pose[:2]
		return world_vertices

	def update(self, force: torch.Tensor, dt: float, contact_pt: torch.Tensor = None):
		"""
		Update the object's state. If a contact point is provided, compute the true torque.
		"""
		# Linear update with damping.
		self.velocity = torch.clamp((self.velocity + (force / M_T) * dt) * LINEAR_DAMPING, 
									min=-MAX_T_VEL, max=MAX_T_VEL)
		self.pose[:2] = self.pose[:2] + self.velocity * dt

		# Compute torque if a contact point is provided.
		if contact_pt is not None:
			r = contact_pt - self.pose[:2]  # lever arm from center of mass
			# 2D cross product (scalar)
			torque = r[0]*force[1] - r[1]*force[0]
		else:
			torque = 0.0

		angular_acceleration = torque / self.moi
		self.angular_velocity = torch.clamp((self.angular_velocity + angular_acceleration * dt) * ANGULAR_DAMPING,
                                           min=-MAX_T_ANG_VEL, max=MAX_T_ANG_VEL)
		self.pose[2] = self.pose[2] + self.angular_velocity * dt

	def draw(self, surface):
		"""
		Draw the T object onto the pygame surface.
		"""
		polygon = self.get_polygon()
		pts = [(int(pt[0].item()), int(pt[1].item())) for pt in polygon]
		pygame.draw.polygon(surface, T_COLOR, pts)
		return polygon

	def compute_contact(self, ee_pos: torch.Tensor) -> (float, torch.Tensor):
		"""
		Compute the minimum distance between the object's edges and the given point (end-effector position).
		Returns the minimum distance and the closest point on the polygon.
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

# ------------------------- Simulation Class -------------------------
class Simulation:
	"""
	Class that encapsulates the simulation logic. This includes handling
	user input, running the main loop, and updating/drawing the arm and object.
	"""
	def __init__(self):
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		pygame.display.set_caption("Push T Data Collection")
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("Arial", 20)

		# Initialize simulation objects.
		self.arm = ArmNR(BASE_POS, LINK_LENGTHS)
		self.T_object = TObject(random_t_pose())
		self.goal_pose = random_t_pose()  # Randomized goal T pose

		# For handling session data.
		self.session_active = False
		self.demo_data = []
		self.session_start_time = None
		self.smoothed_target = None
		self.prev_ee_pos = None

	def run(self):
		"""
		Run the main simulation loop.
		"""
		print("Press [N] to start a new push session.")
		print("Session saves if goal tolerances are met; session quits if IK fails or if the mouse leaves the workspace.")

		running = True
		while running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					if self.demo_data:
						filename = f"session_{int(time.time())}.json"
						with open(filename, "w") as f:
							json.dump(self.demo_data, f, indent=2)
						print(f"Session data saved to {filename}.")
					pygame.quit()
					sys.exit()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_n and not self.session_active:
						# Start a new session: re-randomize both the T object and the goal.
						self.session_active = True
						self.demo_data = []
						self.session_start_time = time.time()
						self.T_object = TObject(random_t_pose())
						self.goal_pose = random_t_pose()
						self.arm.joint_angles = torch.zeros(NUM_JOINTS, dtype=torch.float32)
						self.T_object.velocity = torch.zeros(2, dtype=torch.float32)
						self.T_object.angular_velocity = 0.0
						self.prev_ee_pos = None
						self.smoothed_target = None
						print("New push session started.")

			self.screen.fill(BACKGROUND_COLOR)
			# Draw the goal T as an outline.
			self.draw_goal_T()

			if self.session_active:
				# Get mouse position relative to the arm base.
				mouse_pos = torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32)
				target = mouse_pos

				# If mouse is outside the arm's reachable workspace, cancel the session.
				if torch.norm(target - BASE_POS) > ARM_LENGTH:
					print("Mouse outside arm workspace. Terminating session.")
					self.session_active = False
					self.demo_data = []  # Discard collected data.
					continue

				# Smooth the target if the mouse is near the T object.
				if self.smoothed_target is None:
					self.smoothed_target = target.clone()
				T_poly = self.T_object.get_polygon()
				dist_temp, _ = self.T_object.compute_contact(target)
				if dist_temp < EE_RADIUS:
					delta = target - self.smoothed_target
					max_target_delta = 10.0  # pixels per frame
					delta_clamped = torch.clamp(delta, -max_target_delta, max_target_delta)
					self.smoothed_target += delta_clamped
				else:
					self.smoothed_target = target.clone()

				# Solve IK using the world-coordinate target.
				ik_error = self.arm.solve_ik(self.smoothed_target)
				if not torch.isfinite(torch.tensor(ik_error)) or ik_error > 1e4:
					print("Invalid IK result detected. Terminating session.")
					self.session_active = False
					self.demo_data = []
					continue

				# Compute the world-space end-effector position.
				ee_pos = self.arm.forward_kinematics()  # Already in world coordinates!


				# Compute the end-effector velocity from its previous position.
				if self.prev_ee_pos is None:
					self.prev_ee_pos = ee_pos.clone()
				ee_velocity = ee_pos - self.prev_ee_pos
				self.prev_ee_pos = ee_pos.clone()

				# Check for contact between the end-effector and the T object.
				dist, contact_pt = self.T_object.compute_contact(ee_pos)
				dt = 1.0 / FPS

				if dist < EE_RADIUS:
					raw_push_direction = contact_pt - ee_pos  # Push direction is from EE to T.
					if torch.norm(raw_push_direction) < 1e-3:
						fallback_dir = contact_pt - self.T_object.pose[:2]
						push_direction = fallback_dir / torch.norm(fallback_dir) if torch.norm(fallback_dir) > 0 else torch.tensor([0.0, 0.0])
					else:
						push_direction = raw_push_direction / torch.norm(raw_push_direction)

					# Blend push direction with the EE velocity if they conflict.
					if torch.norm(ee_velocity) > 0:
						ee_dir = ee_velocity / torch.norm(ee_velocity)
						if torch.dot(push_direction, ee_dir) < -0.5:
							blended = 0.7 * ee_dir + 0.3 * push_direction
							if torch.norm(blended) > 0:
								push_direction = blended / torch.norm(blended)

					penetration = EE_RADIUS - dist
					clamped_penetration = min(penetration, MAX_PENETRATION)
					force_magnitude = K_CONTACT * clamped_penetration
					force = force_magnitude * push_direction

					# Hard positional correction if penetration is too large.
					if penetration > MAX_PENETRATION:
						correction = (penetration - MAX_PENETRATION) * push_direction
						self.T_object.pose[:2] += correction

					# Update the T object's state.
					self.T_object.update(force, dt, contact_pt)

					# Save visualization parameters.
					contact_x = int(contact_pt[0].item())
					contact_y = int(contact_pt[1].item())
					arrow_force = force.clone()
					arrow_cp = contact_pt.clone()
				else:
					self.T_object.linear_velocity = torch.zeros(2, dtype=torch.float32)
					self.T_object.angular_velocity = 0.0

				# Draw the arm and the T object.
				self.arm.draw(self.screen)
				self.T_object.draw(self.screen)

				# Draw contact visuals if in contact.
				if dist < EE_RADIUS:
					cross_size = 5
					pygame.draw.line(self.screen, (255, 0, 0),
									 (contact_x - cross_size, contact_y - cross_size),
									 (contact_x + cross_size, contact_y + cross_size), 2)
					pygame.draw.line(self.screen, (255, 0, 0),
									 (contact_x - cross_size, contact_y + cross_size),
									 (contact_x + cross_size, contact_y - cross_size), 2)
					base_arrow_scale = 0.05
					f_norm = torch.norm(arrow_force)
					proposed_length = f_norm * base_arrow_scale
					MAX_ARROW_LENGTH = 60.0
					arrow_scale = MAX_ARROW_LENGTH / f_norm if (proposed_length > MAX_ARROW_LENGTH and f_norm > 0) else base_arrow_scale
					arrow_end_vec = arrow_force * arrow_scale
					arrow_start = (contact_x, contact_y)
					arrow_end = (int((arrow_cp[0] + arrow_end_vec[0]).item()),
								 int((arrow_cp[1] + arrow_end_vec[1]).item()))
					draw_arrow(self.screen, (0, 0, 255), arrow_start, arrow_end, width=3, head_length=10, head_angle=30)

				# Draw the target (mouse) position.
				pygame.draw.circle(self.screen, (0, 200, 0),
				   (int(target[0].item()), int(target[1].item())), 6)


				# Log session data.
				timestamp = time.time() - self.session_start_time
				self.demo_data.append({
					"time": timestamp,
					"mouse_target": target.tolist(),
					"joint_angles": self.arm.joint_angles.tolist(),
					"ee_pos": ee_pos.tolist(),
					"T_pose": self.T_object.pose.tolist(),
					"ik_error": ik_error
				})

				# Check if the T object is within the goal tolerances.
				pos_error = torch.norm(self.T_object.pose[:2] - self.goal_pose[:2]).item()
				orient_error = abs(angle_diff(self.T_object.pose[2], self.goal_pose[2]))
				if pos_error < GOAL_POS_TOL and orient_error < GOAL_ORIENT_TOL:
					print("T object reached desired pose. Session complete.")
					filename = f"session_{int(time.time())}.json"
					with open(filename, "w") as f:
						json.dump(self.demo_data, f, indent=2)
					print(f"Session data saved to {filename}.")
					self.session_active = False

			else:
				# If session is inactive, display instructions.
				text = self.font.render("Press [N] to start a new push session", True, (0, 0, 0))
				self.screen.blit(text, (20, 20))
				self.arm.draw(self.screen)
				self.T_object.draw(self.screen)

			pygame.display.flip()
			self.clock.tick(FPS)

	def draw_goal_T(self):
		"""
		Draw the goal T object as an outline using the current goal_pose.
		"""
		# Compute the polygon for the goal T using the same transformation as TObject.
		theta = self.goal_pose[2]
		cos_t = torch.cos(theta)
		sin_t = torch.sin(theta)
		R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
		rotated = torch.einsum("kj,ij->ki", TObject.local_vertices_adjusted, R)
		world_vertices = rotated + self.goal_pose[:2]
		pts = [(int(pt[0].item()), int(pt[1].item())) for pt in world_vertices]
		pygame.draw.polygon(self.screen, GOAL_T_COLOR, pts, width=3)

# ------------------------- Main Entry Point -------------------------
if __name__ == "__main__":
	sim = Simulation()
	sim.run()

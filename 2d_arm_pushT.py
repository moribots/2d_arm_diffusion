import pygame
import sys
import json
import time
import math
import random

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from scipy.optimize import least_squares

# ------------------------- Global Settings -------------------------
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
BACKGROUND_COLOR = (255, 255, 255)
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)          # Active T color
GOAL_T_COLOR = (0, 200, 0)       # Goal T outline color
FPS = 240

# End-Effector Parameters
EE_RADIUS = 20.0

# Contact Mechanics Parameters
K_CONTACT = 1000.0    # Contact stiffness
M_T = 0.1          # Mass of T block
# Damping constants updated to match gym-pusht (no aggressive clamping)
LINEAR_DAMPING = 1.5
ANGULAR_DAMPING = 1.5
K_DAMPING = 10.0

# Goal tolerances for ending the session
GOAL_POS_TOL = 3.0
GOAL_ORIENT_TOL = 0.08

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
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return diff

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
LINK_LENGTHS = [150.0, 150.0, 150.0, 150.0, 150.0]  # Example link lengths.
NUM_JOINTS = len(LINK_LENGTHS)
ARM_LENGTH = sum(LINK_LENGTHS)

LINK_COLLISION_THRESHOLD = 20.0  
LINK_COLLISION_WEIGHT = 5.0        

class ArmNR:
    """
    A multi-link arm that computes forward kinematics and solves inverse kinematics
    using SciPy's least_squares optimizer.
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
        Compute the (x, y) positions for each joint (including the end-effector).
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
        target_np = np.array(target.numpy())

        def point_to_segment_distance(P, A, B):
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

    def draw(self, surface, color=(50, 100, 200), joint_radius=int(EE_RADIUS), width=5):
        positions = self.compute_joint_positions()
        pts = [(int(pos[0].item()), int(pos[1].item())) for pos in positions]
        for i in range(len(pts) - 1):
            pygame.draw.line(surface, color, pts[i], pts[i+1], width)
        pygame.draw.circle(surface, (0, 0, 0), pts[-1], joint_radius, 2)
        return positions[-1]

# ------------------------- TObject Class -------------------------
def polygon_moi(vertices: torch.Tensor, mass: float) -> float:
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
    centroid = torch.mean(local_vertices, dim=0)
    local_vertices_adjusted = local_vertices - centroid

    def __init__(self, pose: torch.Tensor):
        self.pose = pose.clone()  # [x, y, theta]
        self.velocity = torch.zeros(2, dtype=torch.float32)
        self.angular_velocity = 0.0
        self.moi = polygon_moi(TObject.local_vertices_adjusted, M_T)

    def get_polygon(self):
        theta = self.pose[2]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = torch.einsum("kj,ij->ki", TObject.local_vertices_adjusted, R)
        world_vertices = rotated + self.pose[:2]
        return world_vertices

    def compute_centroid(self):
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
        Update the object's state using standard Euler integration.
        Linear dynamics:
          a = F/m   →   v += a * dt, then apply damping, then update position.
        Angular dynamics:
          torque = r x F (with r = contact_pt - center-of-mass),
          angular acceleration = torque / moi, then update angular velocity and orientation.
        """
        # Linear update
        acceleration = force / M_T
        self.velocity = self.velocity + acceleration * dt
        self.velocity = self.velocity * LINEAR_DAMPING
        self.pose[:2] = self.pose[:2] + self.velocity * dt

        # Angular update using standard cross product (2D)
        if contact_pt is not None:
            true_centroid = self.compute_centroid()
            r = contact_pt - true_centroid
            # Use the velocity-based force for computing torque
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
    Encapsulates the simulation: handling input, running the loop, and updating/drawing the arm and object.
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
        self.goal_pose = random_t_pose()  # Random goal T pose

        self.session_active = False
        self.demo_data = []
        self.session_start_time = None
        self.smoothed_target = None
        self.prev_ee_pos = None

    def run(self):
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

            self.screen.fill(BACKGROUND_COLOR)
            self.draw_goal_T()

            if self.session_active:
                mouse_pos = torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32)
                target = mouse_pos

                if torch.norm(target - BASE_POS) > ARM_LENGTH:
                    print("Mouse outside arm workspace. Terminating session.")
                    self.session_active = False
                    self.demo_data = []
                    continue

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

                ik_error = self.arm.solve_ik(self.smoothed_target)
                if not torch.isfinite(torch.tensor(ik_error)) or ik_error > 1e4:
                    print("Invalid IK result detected. Terminating session.")
                    self.session_active = False
                    self.demo_data = []
                    continue

                ee_pos = self.arm.forward_kinematics()
                dt = 1.0 / FPS
                if self.prev_ee_pos is None:
                    ee_velocity = torch.zeros_like(ee_pos)
                else:
                    ee_velocity = (ee_pos - self.prev_ee_pos) / dt
                self.prev_ee_pos = ee_pos.clone()

                # Check for contact between the end-effector and the T object.
                dist, contact_pt = self.T_object.compute_contact(ee_pos)
                if dist < EE_RADIUS:
                    raw_push_direction = contact_pt - ee_pos
                    if torch.norm(raw_push_direction) < 1e-3:
                        fallback_dir = contact_pt - self.T_object.pose[:2]
                        push_direction = fallback_dir / torch.norm(fallback_dir) if torch.norm(fallback_dir) > 0 else torch.tensor([0.0, 0.0])
                    else:
                        push_direction = raw_push_direction / torch.norm(raw_push_direction)

                    penetration = EE_RADIUS - dist
                    force_magnitude = K_CONTACT * penetration
                    force_from_penetration = force_magnitude * push_direction

                    object_velocity = self.T_object.velocity
                    relative_velocity = ee_velocity - object_velocity
                    velocity_force = K_DAMPING * relative_velocity

                    force = force_from_penetration + velocity_force

                    # Update the T object's state (using the velocity force for torque computation).
                    torque, true_centroid = self.T_object.update(force, dt, contact_pt, velocity_force=velocity_force)

                    contact_x = int(contact_pt[0].item())
                    contact_y = int(contact_pt[1].item())
                else:
                    self.T_object.velocity = torch.zeros(2, dtype=torch.float32)
                    self.T_object.angular_velocity = 0.0

                self.arm.draw(self.screen)
                self.T_object.draw(self.screen)

                if dist < EE_RADIUS:
                    cross_size = 5
                    pygame.draw.line(self.screen, (255, 0, 0),
                                     (contact_x - cross_size, contact_y - cross_size),
                                     (contact_x + cross_size, contact_y + cross_size), 2)
                    pygame.draw.line(self.screen, (255, 0, 0),
                                     (contact_x - cross_size, contact_y + cross_size),
                                     (contact_x + cross_size, contact_y - cross_size), 2)

                    centroid_pt = (int(true_centroid[0].item()), int(true_centroid[1].item()))
                    cp_pt = (int(contact_pt[0].item()), int(contact_pt[1].item()))
                    pygame.draw.line(self.screen, (255, 0, 0), centroid_pt, cp_pt, 3)

                    intermediate_pt = (cp_pt[0], centroid_pt[1])
                    pygame.draw.line(self.screen, (0, 0, 255), centroid_pt, intermediate_pt, 2)
                    pygame.draw.line(self.screen, (0, 0, 255), intermediate_pt, cp_pt, 2)

                    force_scale = 0.1  
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

                pygame.draw.circle(self.screen, (0, 200, 0),
                   (int(target[0].item()), int(target[1].item())), 6)

                timestamp = time.time() - self.session_start_time
                self.demo_data.append({
                    "time": timestamp,
                    "mouse_target": target.tolist(),
                    "joint_angles": self.arm.joint_angles.tolist(),
                    "ee_pos": ee_pos.tolist(),
                    "T_pose": self.T_object.pose.tolist(),
                    "ik_error": ik_error
                })

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
                text = self.font.render("Press [N] to start a new push session", True, (0, 0, 0))
                self.screen.blit(text, (20, 20))
                self.arm.draw(self.screen)
                self.T_object.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(FPS)

    def draw_goal_T(self):
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

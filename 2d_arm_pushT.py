import pygame
import sys
import json
import time
import math
import random

import torch
import torch.nn.functional as F
from einops import rearrange

# --- Environment Settings ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
BACKGROUND_COLOR = (255, 255, 255)
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)          # Active T color
GOAL_T_COLOR = (0, 200, 0)       # Goal T outline color
FPS = 120

# --- 3R Arm Parameters ---
L1 = 300.0
L2 = 300.0
L3 = 200.0
ARM_LENGTH = L1 + L2 + L3

# --- Jacobian IK Parameters ---
IK_ITERATIONS = 100
IK_ALPHA = 0.1
IK_TOL = 1e-3

# --- End-Effector Parameters ---
EE_RADIUS = 16.0

# --- Contact Mechanics Parameters (gym-pusht defaults) ---
K_CONTACT = 1000.0    # Penalty stiffness (softer contact is achieved by adjusting this)
M_T = 100.0           # Mass of T block
I_MOMENT = 10.0       # Moment of inertia for rotation
ANGULAR_DAMPING = 0.1
LINEAR_DAMPING = 0.1

# Maximum allowed penetration (to avoid huge forces)
MAX_PENETRATION = 0.5

# Maximum velocities (for clamping)
MAX_T_VEL = 100.0 # m/s
MAX_T_ANG_VEL = 10.0 # rad/s

# --- Function to Randomize T Pose ---
def random_t_pose():
    # Use a margin to keep the T completely visible on screen.
    margin = 100  
    x = random.uniform(margin, SCREEN_WIDTH - margin)
    y = random.uniform(margin, SCREEN_HEIGHT - margin)
    theta = random.uniform(-math.pi, math.pi)
    return torch.tensor([x, y, theta], dtype=torch.float32)

# --- Goal Settings ---
DESIRED_T_POSE = random_t_pose()  # Randomized goal pose
GOAL_POS_TOL = 2.0
GOAL_ORIENT_TOL = 0.2

# --- Arm Base Position ---
BASE_POS = torch.tensor([300.0, 300.0], dtype=torch.float32)

# --- T-Shape Object Definition ---
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

def get_t_polygon(pose: torch.Tensor) -> torch.Tensor:
    """Return the world-space vertices of the T-shaped block given its pose (x, y, theta)."""
    theta = pose[2]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = torch.einsum("kj,ij->ki", local_vertices_adjusted, R)
    world_vertices = rotated + pose[:2]
    return world_vertices

def compute_contact(ee_pos: torch.Tensor, polygon: torch.Tensor) -> (float, torch.Tensor):
    """Compute the minimum distance between ee_pos and polygon edges, returning the distance and closest point."""
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

def angle_diff(a: float, b: float) -> float:
    """Return the difference between two angles (in radians), wrapped to [-pi, pi]."""
    diff = (a - b + torch.pi) % (2 * torch.pi) - torch.pi
    return diff.item()

# --- Contact Mechanics: Penalty Force Model ---
def compute_contact_force(penetration: float, k_contact: float = K_CONTACT) -> float:
    """Compute force magnitude using a penalty model."""
    return k_contact * penetration if penetration > 0 else 0.0

# --- Kinematics Functions using torch and einsum ---
def forward_kinematics(joint_angles: torch.Tensor) -> torch.Tensor:
    theta1, theta2, theta3 = joint_angles[0], joint_angles[1], joint_angles[2]
    x = L1 * torch.cos(theta1) + L2 * torch.cos(theta1 + theta2) + L3 * torch.cos(theta1 + theta2 + theta3)
    y = L1 * torch.sin(theta1) + L2 * torch.sin(theta1 + theta2) + L3 * torch.sin(theta1 + theta2 + theta3)
    return torch.stack([x, y], dim=0)

def compute_jacobian(joint_angles: torch.Tensor) -> torch.Tensor:
    theta1, theta2, theta3 = joint_angles[0], joint_angles[1], joint_angles[2]
    t12 = theta1 + theta2
    t123 = theta1 + theta2 + theta3
    dx_dtheta1 = -L1 * torch.sin(theta1) - L2 * torch.sin(t12) - L3 * torch.sin(t123)
    dx_dtheta2 = -L2 * torch.sin(t12) - L3 * torch.sin(t123)
    dx_dtheta3 = -L3 * torch.sin(t123)
    dy_dtheta1 = L1 * torch.cos(theta1) + L2 * torch.cos(t12) + L3 * torch.cos(t123)
    dy_dtheta2 = L2 * torch.cos(t12) + L3 * torch.cos(t123)
    dy_dtheta3 = L3 * torch.cos(t123)
    J_flat = torch.stack([dx_dtheta1, dx_dtheta2, dx_dtheta3,
                           dy_dtheta1, dy_dtheta2, dy_dtheta3], dim=0)
    J = rearrange(J_flat, '(a b) -> a b', a=2)
    return J

def ik_jacobian(target: torch.Tensor, initial_angles: torch.Tensor, iterations=IK_ITERATIONS, alpha=IK_ALPHA, tol=IK_TOL):
    theta = initial_angles.clone()
    for i in range(iterations):
        pos = forward_kinematics(theta)
        error = target - pos
        if not torch.isfinite(torch.norm(error)) or torch.norm(error) > 1e4:
            print("Invalid IK result detected. Quitting session.")
            return theta, float('inf')
        if torch.norm(error) < tol:
            break
        J = compute_jacobian(theta)
        J_pinv = torch.pinverse(J)
        delta_theta = alpha * torch.einsum("ij,j->i", J_pinv, error)
        theta = theta + delta_theta
    return theta, torch.norm(error).item()

# --- Drawing Functions ---
def draw_arm(surface, base: torch.Tensor, joint_angles: torch.Tensor):
    theta1, theta2, theta3 = joint_angles[0], joint_angles[1], joint_angles[2]
    x0, y0 = base[0].item(), base[1].item()
    x1 = x0 + L1 * torch.cos(theta1).item()
    y1 = y0 + L1 * torch.sin(theta1).item()
    x2 = x1 + L2 * torch.cos(theta1 + theta2).item()
    y2 = y1 + L2 * torch.sin(theta1 + theta2).item()
    x3 = x2 + L3 * torch.cos(theta1 + theta2 + theta3).item()
    y3 = y2 + L3 * torch.sin(theta1 + theta2 + theta3).item()
    pygame.draw.line(surface, ARM_COLOR, (x0, y0), (x1, y1), 5)
    pygame.draw.line(surface, ARM_COLOR, (x1, y1), (x2, y2), 5)
    pygame.draw.line(surface, ARM_COLOR, (x2, y2), (x3, y3), 5)
    for pt in [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]:
        pygame.draw.circle(surface, (0, 0, 0), (int(pt[0]), int(pt[1])), 16, 2)
    return torch.tensor([x3, y3], dtype=torch.float32)

def draw_T(surface, pose: torch.Tensor):
    polygon = get_t_polygon(pose)
    pts = [(int(pt[0].item()), int(pt[1].item())) for pt in polygon]
    pygame.draw.polygon(surface, T_COLOR, pts)
    return polygon

def draw_goal_T(surface):
    polygon = get_t_polygon(DESIRED_T_POSE)
    pts = [(int(pt[0].item()), int(pt[1].item())) for pt in polygon]
    pygame.draw.polygon(surface, GOAL_T_COLOR, pts, width=3)

def draw_arrow(surface, color, start, end, width=3, head_length=10, head_angle=30):
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

# --- History-based Positional Correction ---
prev_ee_pos = None  # For storing previous EE position

# --- Main Loop ---
def main():
    global prev_ee_pos, smoothed_target, DESIRED_T_POSE
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Push T Data Collection (History-based Correction)")
    clock = pygame.time.Clock()
    
    print("Press [N] to start a new push session.")
    print("Session will save if goal tolerances are met; session quits if an invalid IK result is detected or if the mouse is outside the arm workspace.")

    session_active = False
    demo_data = []
    session_start_time = None
    smoothed_target = None

    current_angles = torch.zeros(3, dtype=torch.float32)
    # Randomize starting T pose on launch. (It will be re-randomized on each new session.)
    T_pose = random_t_pose()
    T_velocity = torch.zeros(2, dtype=torch.float32)
    T_angular_velocity = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if demo_data:
                    filename = f"session_{int(time.time())}.json"
                    with open(filename, "w") as f:
                        json.dump(demo_data, f, indent=2)
                    print(f"Session data saved to {filename}.")
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n and not session_active:
                    session_active = True
                    demo_data = []
                    session_start_time = time.time()
                    # Randomize both start and goal T poses for new session.
                    T_pose = random_t_pose()
                    DESIRED_T_POSE = random_t_pose()
                    current_angles = torch.zeros(3, dtype=torch.float32)
                    T_velocity = torch.zeros(2, dtype=torch.float32)
                    T_angular_velocity = 0.0
                    prev_ee_pos = None
                    smoothed_target = None
                    print("New push session started.")
        
        screen.fill(BACKGROUND_COLOR)
        draw_goal_T(screen)
        
        if session_active:
            mouse_pos = torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32)
            relative_target = mouse_pos - BASE_POS
            
            # Terminate session if mouse is outside the arm workspace.
            if torch.norm(relative_target) > ARM_LENGTH:
                print("Mouse outside arm workspace. Terminating session.")
                session_active = False
                demo_data = []  # Discard demo data.
                continue

            # Initialize smoothed_target if needed.
            if smoothed_target is None:
                smoothed_target = relative_target.clone()
            # Apply smoothing if mouse is near T (within 4*EE_RADIUS)
            T_polygon = get_t_polygon(T_pose)
            dist_temp, _ = compute_contact(relative_target, T_polygon)
            if dist_temp < EE_RADIUS * 4.0:
                delta = relative_target - smoothed_target
                max_target_delta = 1.0  # pixels per frame
                delta_clamped = torch.clamp(delta, -max_target_delta, max_target_delta)
                smoothed_target = smoothed_target + delta_clamped
            else:
                smoothed_target = relative_target.clone()

            new_angles, err_norm = ik_jacobian(smoothed_target, current_angles)
            # If IK result is invalid, terminate the session.
            if not torch.isfinite(torch.tensor(err_norm)) or err_norm > 1e4:
                print("Invalid IK result detected. Terminating session.")
                session_active = False
                demo_data = []
                continue

            current_angles = new_angles
            ee_pos = forward_kinematics(current_angles) + BASE_POS

            # Maintain EE history for positional correction.
            if prev_ee_pos is None:
                prev_ee_pos = ee_pos.clone()
            ee_velocity = ee_pos - prev_ee_pos
            prev_ee_pos = ee_pos.clone()

            T_polygon = get_t_polygon(T_pose)
            dist, contact_pt = compute_contact(ee_pos, T_polygon)
            dt = 1.0 / FPS

            if dist < EE_RADIUS:
                raw_push_direction = contact_pt - ee_pos  # push T away from EE
                if torch.norm(raw_push_direction) < 1e-3:
                    fallback_dir = contact_pt - T_pose[:2]
                    push_direction = fallback_dir / torch.norm(fallback_dir) if torch.norm(fallback_dir) > 0 else torch.tensor([0.0, 0.0])
                else:
                    push_direction = raw_push_direction / torch.norm(raw_push_direction)
                
                # History-based blending if push_direction conflicts with ee_velocity.
                if torch.norm(ee_velocity) > 0:
                    ee_dir = ee_velocity / torch.norm(ee_velocity)
                    if torch.dot(push_direction, ee_dir) < -0.5:
                        blended = 0.7 * ee_dir + 0.3 * push_direction
                        if torch.norm(blended) > 0:
                            push_direction = blended / torch.norm(blended)
                
                penetration = EE_RADIUS - dist
                clamped_penetration = min(penetration, MAX_PENETRATION)
                force_magnitude = compute_contact_force(clamped_penetration)
                force = force_magnitude * push_direction

                # Hard positional correction if penetration exceeds threshold.
                if penetration > MAX_PENETRATION:
                    correction = (penetration - MAX_PENETRATION) * push_direction
                    T_pose[:2] = T_pose[:2] + correction

                T_velocity = torch.clamp((T_velocity + (force / M_T) * dt) * LINEAR_DAMPING, min=-MAX_T_VEL, max=MAX_T_VEL)
                T_pose[:2] = T_pose[:2] + T_velocity * dt

                r = contact_pt - T_pose[:2]
                torque = r[0] * force[1] - r[1] * force[0]
                angular_acceleration = torque / I_MOMENT
                T_angular_velocity = torch.clamp((T_angular_velocity + angular_acceleration * dt) * ANGULAR_DAMPING, min=-MAX_T_ANG_VEL, max=MAX_T_ANG_VEL)
                T_pose[2] = T_pose[2] + T_angular_velocity * dt

                # Save visualization parameters for drawing after everything:
                contact_x = int(contact_pt[0].item())
                contact_y = int(contact_pt[1].item())
                arrow_force = force.clone()
                arrow_cp = contact_pt.clone()

            draw_arm(screen, BASE_POS, current_angles)
            draw_T(screen, T_pose)

            # Draw contact visuals on top:
            if dist < EE_RADIUS:
                cross_size = 5
                pygame.draw.line(screen, (255, 0, 0),
                                 (contact_x - cross_size, contact_y - cross_size),
                                 (contact_x + cross_size, contact_y + cross_size), 2)
                pygame.draw.line(screen, (255, 0, 0),
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
                draw_arrow(screen, (0, 0, 255), arrow_start, arrow_end, width=3, head_length=10, head_angle=30)
            
            target_screen = BASE_POS + relative_target
            pygame.draw.circle(screen, (0, 200, 0),
                               (int(target_screen[0].item()), int(target_screen[1].item())), 6)
            
            timestamp = time.time() - session_start_time
            demo_data.append({
                "time": timestamp,
                "mouse_target": relative_target.tolist(),
                "joint_angles": current_angles.tolist(),
                "ee_pos": ee_pos.tolist(),
                "T_pose": T_pose.tolist(),
                "ik_error": err_norm
            })
            
            pos_error = torch.norm(T_pose[:2] - DESIRED_T_POSE[:2]).item()
            orient_error = abs(angle_diff(T_pose[2], DESIRED_T_POSE[2]))
            if pos_error < GOAL_POS_TOL and orient_error < GOAL_ORIENT_TOL:
                print("T object reached desired pose. Session complete.")
                filename = f"session_{int(time.time())}.json"
                with open(filename, "w") as f:
                    json.dump(demo_data, f, indent=2)
                print(f"Session data saved to {filename}.")
                session_active = False
        else:
            font = pygame.font.SysFont("Arial", 20)
            text = font.render("Press [N] to start a new push session", True, (0, 0, 0))
            screen.blit(text, (20, 20))
            draw_arm(screen, BASE_POS, current_angles)
            draw_T(screen, T_pose)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

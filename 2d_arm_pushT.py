import pygame
import sys
import json
import time

import torch
import torch.nn.functional as F
from einops import rearrange

# --- Settings ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)            # Color for the active (pushed) T
GOAL_T_COLOR = (0, 200, 0)         # Outline color for the goal T shape
INVALID_TARGET_COLOR = (255, 0, 0)
PATH_COLOR = (0, 0, 255)
FPS = 60

# 3R Arm parameters (all lengths in pixels)
L1 = 100.0  # First link
L2 = 100.0  # Second link
L3 = 80.0   # Third link
ARM_LENGTH = L1 + L2 + L3

# Jacobian IK parameters
IK_ITERATIONS = 100
IK_ALPHA = 0.1
IK_TOL = 1e-3

# EE parameters
EE_RADIUS = 8.0  # End-effector “tool” radius

# Pushing simulation parameters
PUSH_SPEED = 2.0       # Base translation increment per frame
TORQUE_FACTOR = 0.005  # Factor to convert moment to angular change

# Goal (desired T pose): (x, y, theta)
DESIRED_T_POSE = torch.tensor([500.0, 500.0, 0.0], dtype=torch.float32)
GOAL_POS_TOL = 30.0      # Position tolerance (pixels)
GOAL_ORIENT_TOL = 0.2    # Orientation tolerance (radians)

# Arm base position (as a torch tensor)
BASE_POS = torch.tensor([300.0, 300.0], dtype=torch.float32)

# --- T-Shape Object Definition ---
# Define a T-shaped polygon (in local coordinates)
local_vertices = torch.tensor([
    [-40.0, -10.0],  # top-left of crossbar
    [40.0, -10.0],   # top-right of crossbar
    [40.0, 10.0],    # bottom-right of crossbar
    [10.0, 10.0],    # top-right of stem (at crossbar bottom)
    [10.0, 70.0],    # bottom-right of stem
    [-10.0, 70.0],   # bottom-left of stem
    [-10.0, 10.0],   # top-left of stem
    [-40.0, 10.0]    # bottom-left of crossbar
], dtype=torch.float32)

# Scale the T to make it larger
T_SCALE = 2.0
local_vertices = local_vertices * T_SCALE

# Compute the approximate centroid and shift vertices so that the object’s reference is at (0,0)
centroid = torch.mean(local_vertices, dim=0)
local_vertices_adjusted = local_vertices - centroid

def get_t_polygon(pose: torch.Tensor) -> torch.Tensor:
    theta = pose[2]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = torch.einsum("kj,ij->ki", local_vertices_adjusted, R)
    world_vertices = rotated + pose[:2]
    return world_vertices

def compute_contact(ee_pos: torch.Tensor, polygon: torch.Tensor) -> (float, torch.Tensor):
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
    diff = (a - b + torch.pi) % (2 * torch.pi) - torch.pi
    return diff.item()

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

def ik_jacobian(target: torch.Tensor, initial_angles: torch.Tensor, 
                iterations=IK_ITERATIONS, alpha=IK_ALPHA, tol=IK_TOL):
    theta = initial_angles.clone()
    for i in range(iterations):
        pos = forward_kinematics(theta)
        error = target - pos
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
        pygame.draw.circle(surface, (0, 0, 0), (int(pt[0]), int(pt[1])), 8)
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

# --- Main Data Collection with Sessions ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Push T Data Collection (Realistic Pushing with Einsum)")
    clock = pygame.time.Clock()
    
    print("Press [N] to start a new push session.")
    print("Use the mouse to set the desired EE target (relative to base).")
    print("Session ends if the target is invalid or if the T object reaches the desired 2D pose.")
    
    session_active = False
    demo_data = []  # One session (rollout) of data
    session_start_time = None

    current_angles = torch.zeros(3, dtype=torch.float32)
    T_pose = torch.tensor([250.0, 250.0, 0.0], dtype=torch.float32)
    
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
                    T_pose = torch.tensor([250.0, 250.0, 0.0], dtype=torch.float32)
                    current_angles = torch.zeros(3, dtype=torch.float32)
                    print("New push session started.")
        
        screen.fill(BACKGROUND_COLOR)
        draw_goal_T(screen)
        
        if session_active:
            mouse_pos = torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32)
            relative_target = mouse_pos - BASE_POS
            
            if torch.norm(relative_target) > ARM_LENGTH:
                print("Invalid IK target produced. Ending session.")
                session_active = False
                filename = f"session_{int(time.time())}.json"
                with open(filename, "w") as f:
                    json.dump(demo_data, f, indent=2)
                print(f"Session data saved to {filename}.")
                continue

            new_angles, err_norm = ik_jacobian(relative_target, current_angles)
            current_angles = new_angles
            ee_pos = forward_kinematics(current_angles) + BASE_POS

            T_polygon = get_t_polygon(T_pose)
            dist, contact_pt = compute_contact(ee_pos, T_polygon)
            if dist < EE_RADIUS:
                # Compute raw push direction from contact point to EE.
                raw_push_direction = ee_pos - contact_pt
                if torch.norm(raw_push_direction) < 1e-3:
                    # Fallback if EE is deeply penetrating: use direction from T center to EE.
                    fallback_dir = ee_pos - T_pose[:2]
                    if torch.norm(fallback_dir) > 0:
                        push_direction = fallback_dir / torch.norm(fallback_dir)
                    else:
                        push_direction = torch.tensor([0.0, 0.0])
                else:
                    push_direction = raw_push_direction / torch.norm(raw_push_direction)
                # Compute penetration depth.
                penetration = EE_RADIUS - dist
                # Update T position by a displacement proportional to PUSH_SPEED plus penetration.
                displacement = (PUSH_SPEED + penetration) * push_direction
                T_pose[:2] = T_pose[:2] + displacement
                # Compute lever arm and update orientation.
                r = contact_pt - T_pose[:2]
                torque = r[0] * push_direction[1] - r[1] * push_direction[0]
                T_pose[2] = T_pose[2] + TORQUE_FACTOR * torque

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
                session_active = False
                filename = f"session_{int(time.time())}.json"
                with open(filename, "w") as f:
                    json.dump(demo_data, f, indent=2)
                print(f"Session data saved to {filename}.")

            target_screen = BASE_POS + relative_target
            pygame.draw.circle(screen, (0, 200, 0), (int(target_screen[0].item()), int(target_screen[1].item())), 6)
            draw_arm(screen, BASE_POS, current_angles)
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

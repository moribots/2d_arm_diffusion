import pygame
import sys
import json
import time

import torch
from torch import tensor
import torch.nn.functional as F

from einops import rearrange  # for tensor reshaping

# --- Settings ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)
INVALID_TARGET_COLOR = (255, 0, 0)
PATH_COLOR = (0, 0, 255)
FPS = 60

# 3R Arm parameters (all lengths in pixels)
L1 = 100.0  # First link length
L2 = 100.0  # Second link length
L3 = 80.0   # Third link length
ARM_LENGTH = L1 + L2 + L3  # Maximum reach

# Jacobian IK parameters
IK_ITERATIONS = 100
IK_ALPHA = 0.1
IK_TOL = 1e-3

# Pushing simulation parameters
PUSH_THRESHOLD = 20.0  # distance threshold for contact between EE and T
PUSH_SPEED = 2.0       # how much T moves per frame when in contact

# Desired region for T (e.g., a circle where we want T to end up)
DESIRED_T_POS = torch.tensor([500.0, 500.0], dtype=torch.float32)
DESIRED_T_TOL = 30.0  # tolerance radius (in pixels)

# Arm base position (as a torch tensor)
BASE_POS = torch.tensor([300.0, 300.0], dtype=torch.float32)

# --- Kinematics Functions using torch and einops ---
def forward_kinematics(joint_angles: torch.Tensor) -> torch.Tensor:
    """
    Computes the EE position (relative to BASE_POS) for a 3R planar arm.
    joint_angles: tensor of shape (3,) [theta1, theta2, theta3] in radians.
    Returns: tensor([x, y]) for the end-effector position.
    """
    theta1, theta2, theta3 = joint_angles[0], joint_angles[1], joint_angles[2]
    x = L1 * torch.cos(theta1) + L2 * torch.cos(theta1 + theta2) + L3 * torch.cos(theta1 + theta2 + theta3)
    y = L1 * torch.sin(theta1) + L2 * torch.sin(theta1 + theta2) + L3 * torch.sin(theta1 + theta2 + theta3)
    return torch.stack([x, y], dim=0)

def compute_jacobian(joint_angles: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2x3 Jacobian matrix for the 3R planar arm at the given joint angles.
    """
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
    J = rearrange(J_flat, '(a b) -> a b', a=2)  # shape (2,3)
    return J

def ik_jacobian(target: torch.Tensor, initial_angles: torch.Tensor, 
                iterations=IK_ITERATIONS, alpha=IK_ALPHA, tol=IK_TOL):
    """
    Uses a Jacobian-based iterative method to solve the IK problem.
    target: desired EE position (relative to BASE_POS) as a tensor of shape (2,)
    initial_angles: initial guess for joint angles (tensor of shape (3,))
    Returns: joint_angles (tensor of shape (3,)) and final error norm (a float).
    """
    theta = initial_angles.clone()
    for i in range(iterations):
        pos = forward_kinematics(theta)
        error = target - pos
        if torch.norm(error) < tol:
            break
        J = compute_jacobian(theta)
        J_pinv = torch.pinverse(J)
        delta_theta = alpha * J_pinv @ error
        theta = theta + delta_theta
    return theta, torch.norm(error).item()

# --- Drawing Functions ---
def draw_arm(surface, base: torch.Tensor, joint_angles: torch.Tensor):
    """
    Draws the 3R arm on the given surface.
    Returns the end-effector (EE) screen position as a torch tensor.
    """
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

def draw_T(surface, pos: torch.Tensor):
    """
    Draws the T-shaped object as a rectangle at the given screen position.
    """
    rect = pygame.Rect(0, 0, 60, 60)
    rect.center = (int(pos[0].item()), int(pos[1].item()))
    pygame.draw.rect(surface, T_COLOR, rect)
    return rect

def draw_desired_region(surface):
    """
    Draws the desired target region for the T object.
    """
    pygame.draw.circle(surface, (0, 200, 0), (int(DESIRED_T_POS[0].item()), int(DESIRED_T_POS[1].item())), int(DESIRED_T_TOL), 2)

# --- Main Data Collection with Sessions using Jacobian IK and Einops ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Push T Data Collection")
    clock = pygame.time.Clock()
    
    print("Press [N] to start a new push session.")
    print("Use the mouse to set the desired EE target (relative to base).")
    print("Session ends if the target is invalid (outside reachable area) or if T is pushed into the desired region.")

    session_active = False
    demo_data = []  # List to hold data for one session (rollout)
    session_start_time = None

    # Initialize the arm's joint angles as a torch tensor (3,)
    current_angles = torch.zeros(3, dtype=torch.float32)
    
    # Initialize T object at a starting position (as torch tensor)
    T_pos = torch.tensor([250.0, 250.0], dtype=torch.float32)

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
                    # Start a new session
                    session_active = True
                    demo_data = []
                    session_start_time = time.time()
                    T_pos = torch.tensor([250.0, 250.0], dtype=torch.float32)
                    current_angles = torch.zeros(3, dtype=torch.float32)
                    print("New push session started.")
        
        screen.fill(BACKGROUND_COLOR)
        draw_desired_region(screen)
        
        if session_active:
            # Get mouse position as torch tensor (relative to screen)
            mouse_pos = torch.tensor(pygame.mouse.get_pos(), dtype=torch.float32)
            relative_target = mouse_pos - BASE_POS  # desired EE target relative to base

            # Check feasibility: target must be within reachable workspace.
            if torch.norm(relative_target) > ARM_LENGTH:
                print("Invalid IK target produced. Ending session.")
                session_active = False
                filename = f"session_{int(time.time())}.json"
                with open(filename, "w") as f:
                    json.dump(demo_data, f, indent=2)
                print(f"Session data saved to {filename}.")
                continue

            # Use Jacobian-based IK to compute joint angles for the target.
            new_angles, err_norm = ik_jacobian(relative_target, current_angles)
            current_angles = new_angles  # update the current angles

            # Compute the current EE position (absolute screen position)
            ee_pos = forward_kinematics(current_angles) + BASE_POS

            # Simulate pushing:
            # If the EE is near the T object, push T in the direction from T to EE.
            if torch.norm(ee_pos - T_pos) <= PUSH_THRESHOLD:
                push_direction = ee_pos - T_pos
                if torch.norm(push_direction) > 0:
                    push_direction = push_direction / torch.norm(push_direction)
                T_pos = T_pos + PUSH_SPEED * push_direction

            # Record data for this time step.
            timestamp = time.time() - session_start_time
            demo_data.append({
                "time": timestamp,
                "mouse_target": relative_target.tolist(),
                "joint_angles": current_angles.tolist(),
                "ee_pos": ee_pos.tolist(),
                "T_pos": T_pos.tolist(),
                "ik_error": err_norm
            })

            # Check if T is within the desired region.
            if torch.norm(T_pos - DESIRED_T_POS) < DESIRED_T_TOL:
                print("T object pushed into desired region. Session complete.")
                session_active = False
                filename = f"session_{int(time.time())}.json"
                with open(filename, "w") as f:
                    json.dump(demo_data, f, indent=2)
                print(f"Session data saved to {filename}.")

            # Draw the desired EE target marker (green if valid)
            target_color = (0, 200, 0)
            target_screen = BASE_POS + relative_target
            pygame.draw.circle(screen, target_color, (int(target_screen[0].item()), int(target_screen[1].item())), 6)

            # Draw arm based on current joint angles.
            draw_arm(screen, BASE_POS, current_angles)
        else:
            # When no session is active, display instructions.
            font = pygame.font.SysFont("Arial", 20)
            text = font.render("Press [N] to start a new push session", True, (0, 0, 0))
            screen.blit(text, (20, 20))
            draw_arm(screen, BASE_POS, current_angles)
        
        # Draw the T object.
        draw_T(screen, T_pos)
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

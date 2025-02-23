import torch

# Screen settings
SCREEN_WIDTH = 672
SCREEN_HEIGHT = 672
BACKGROUND_COLOR = (255, 255, 255)

# Colors
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)         # Color for the object
GOAL_T_COLOR = (0, 200, 0)      # Outline color for the goal T

# End-effector parameters
EE_RADIUS = 20.0

# Arm parameters
LINK_LENGTHS = [150.0, 150.0, 150.0, 150.0, 150.0]
NUM_JOINTS = len(LINK_LENGTHS)
ARM_LENGTH = sum(LINK_LENGTHS)
BASE_POS = torch.tensor([300.0, 300.0], dtype=torch.float32)

# Contact mechanics parameters
K_CONTACT = 800.0    # Contact stiffness
M_T = 0.8             # Mass of object
LINEAR_DAMPING = 1.2  # Linear velocity damping factor
ANGULAR_DAMPING = 1.0 # Angular velocity damping factor
K_DAMPING = 10.0      # Damping constant for velocity differences

# Goal tolerances for ending the session
GOAL_POS_TOL = 3.0
GOAL_ORIENT_TOL = 0.08

# Arm collision parameters
LINK_COLLISION_THRESHOLD = 20.0  
LINK_COLLISION_WEIGHT = 5.0

# Training parameters for diffusion policy
T = 1000            # Total number of diffusion timesteps
BATCH_SIZE = 256     # Batch size used during training
EPOCHS = 5000         # Total number of epochs for training
LEARNING_RATE = 1e-3  # Learning rate for the optimizer

# Set the temporal window size.
WINDOW_SIZE = 14  # Number of consecutive action timestamps to form the sequence.

# Action and condition dimensions
ACTION_DIM = 2      # EE position is 2D
CONDITION_DIM = 15          # 6 steps * 2.5 params/step (x,y,theta) + 3 goal params

# Temporal Parameters
FPS = 100
SEC_PER_SAMPLE = 0.1
DEMO_DATA_FRAMES = 16
IMG_RES = 224

# Directory for saving training data
TRAINING_DATA_DIR = "training_data"

"""
Configuration constants and image transformation pipelines for the diffusion policy project.
"""

import torch
import torchvision.transforms as transforms
import random
import numpy as np

# Set random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)

# Screen and environment settings.
SCREEN_WIDTH = 680
SCREEN_HEIGHT = 680
ACTION_LIM = 512.0  # Scale factor for actions from pushT (see: https://github.com/huggingface/gym-pusht)
BACKGROUND_COLOR = (255, 255, 255)

# Define colors.
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)         # Color for the object (T)
GOAL_T_COLOR = (0, 200, 0)      # Outline color for the goal T

# End-effector parameters.
EE_RADIUS = 20.0

# Arm parameters.
LINK_LENGTHS = [150.0, 150.0, 150.0, 150.0, 150.0]
INITIAL_ANGLES = [0.29430179, -0.75699092, -1.44649063, -1.0642895, -0.49411484]
NUM_JOINTS = len(LINK_LENGTHS)
ARM_LENGTH = sum(LINK_LENGTHS)
BASE_POS = torch.tensor([300.0, 300.0], dtype=torch.float32)

# Contact mechanics parameters.
K_CONTACT = 800.0    # Contact stiffness
M_T = 0.8             # Mass of object
LINEAR_DAMPING = 1.2  # Damping factor for linear velocity
ANGULAR_DAMPING = 1.0 # Damping factor for angular velocity
K_DAMPING = 10.0      # Damping constant for velocity differences

# Goal tolerances.
GOAL_POS_TOL = 3.0
GOAL_ORIENT_TOL = 0.08

# Arm collision parameters.
LINK_COLLISION_THRESHOLD = 20.0
LINK_COLLISION_WEIGHT = 5.0

# Training parameters.
T = 1000
BATCH_SIZE = 832
EPOCHS = 900
# Settings from: https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/diffusion/configuration_diffusion.py#L154-L160
OPTIMIZER_LR = 1e-4
OPTIMIZER_BETAS = (0.95, 0.999)
OPTIMIZER_EPS = 1e-8
OPTIMIZER_WEIGHT_DECAY = 1e-6
SCHEDULER_NAME = "cosine"
SCHEDULER_WARMUP_STEPS = 0

# Temporal window size for action sequence.
WINDOW_SIZE = 14

# Visual encoder output: each image now yields 64 features.
IMAGE_FEATURE_DIM = 64

# Conditioning dimension: agent state (4) + features from two images (2*64 = 128) = 132.
CONDITION_DIM = 4 + 2 * IMAGE_FEATURE_DIM
ACTION_DIM = 2

# Temporal parameters.
FPS = 100
SEC_PER_SAMPLE = 0.1
DEMO_DATA_FRAMES = 16
IMG_RES = 96

# Image transformation pipeline (resizing, converting to tensor, and normalizing).
image_transform = transforms.Compose([
	transforms.Resize((IMG_RES, IMG_RES)),  # Resize image to IMG_RES x IMG_RES.
	transforms.ToTensor(),                  # Convert PIL Image to Tensor with values [0,1].
	transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats.
						 std=[0.229, 0.224, 0.225]),
])

# Directories.
TRAINING_DATA_DIR = "training_data"
OUTPUT_DIR = ""

# Diffusion loss masking configuration.
DO_MASK_LOSS_FOR_PADDING = True

# Dataset and environment type settings.
DATASET_TYPE = "lerobot"      # Options: "custom", "lerobot"
LE_ROBOT_GYM_ENV_NAME = "gym_pusht/PushT-v0"

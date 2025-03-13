"""
Configuration constants and image transformation pipelines for the diffusion policy project.
"""

import torch
import torchvision.transforms as transforms
import numpy as np

# Screen and environment settings.
SCREEN_WIDTH = 680
SCREEN_HEIGHT = 680
ACTION_LIM = 512.0  # Scale factor for actions.
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
BATCH_SIZE_MULT = 1
BATCH_SIZE = 64 * BATCH_SIZE_MULT
EPOCHS = 500
VALIDATION_INTERVAL = 50
OPTIMIZER_LR = 1e-4 * np.sqrt(BATCH_SIZE_MULT)
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_EPS = 1e-8
OPTIMIZER_WEIGHT_DECAY = 1e-6
SCHEDULER_WARMUP_EPOCHS = 5

# Temporal window size for action sequence.
WINDOW_SIZE = 16

# Visual encoder output.
IMAGE_FEATURE_DIM = 64

# Conditioning dimension.
CONDITION_DIM = 2
ACTION_DIM = 2

# Temporal parameters.
FPS = 100
SEC_PER_SAMPLE = 0.1
DEMO_DATA_FRAMES = 16
IMG_RES = 96

# Image transformation pipeline.
image_transform = transforms.Compose([
	transforms.Resize((IMG_RES, IMG_RES)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						 std=[0.229, 0.224, 0.225]),
])

# Directories.
TRAINING_DATA_DIR = "training_data"
OUTPUT_DIR = ""  # e.g., OUTPUT_DIR = "/kaggle/working/"
DATA_SOURCE_DIR = "data"  # Root directory for all data sources

# For normalization stats and model checkpoints
NORM_STATS_FILENAME = "normalization_stats.parquet"
MODEL_CHECKPOINT_FILENAME = "diffusion_policy.pth"

# Diffusion loss masking configuration.
DO_MASK_LOSS_FOR_PADDING = True
DO_SQRT_ALPHA_BAR_WEIGHTING = False

# Dataset and environment type settings.
DATASET_TYPE = "lerobot"      # Options: "custom", "lerobot"
LE_ROBOT_GYM_ENV_NAME = "gym_pusht/PushT-v0"

def print_settings():
	"""
	Print the current configuration settings.
	"""
	settings = {
		'Screen Settings': {
			'Width': SCREEN_WIDTH,
			'Height': SCREEN_HEIGHT,
			'Action Limit': ACTION_LIM,
			'Background Color': BACKGROUND_COLOR
		},
		'Colors': {
			'Arm': ARM_COLOR,
			'T Object': T_COLOR,
			'Goal T': GOAL_T_COLOR
		},
		'Arm Parameters': {
			'End Effector Radius': EE_RADIUS,
			'Link Lengths': LINK_LENGTHS,
			'Initial Angles': INITIAL_ANGLES,
			'Number of Joints': NUM_JOINTS,
			'Total Arm Length': ARM_LENGTH,
			'Base Position': BASE_POS.tolist()
		},
		'Physics Parameters': {
			'Contact Stiffness': K_CONTACT,
			'Object Mass': M_T,
			'Linear Damping': LINEAR_DAMPING,
			'Angular Damping': ANGULAR_DAMPING,
			'Damping Constant': K_DAMPING
		},
		'Training Parameters': {
			'T': T,
			'Batch Size': BATCH_SIZE,
			'Epochs': EPOCHS,
			'Validation Interval': VALIDATION_INTERVAL,
			'Optimizer LR': OPTIMIZER_LR
		}
	}
	print(settings)

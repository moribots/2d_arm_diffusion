import torch
import torchvision.transforms as transforms  # new import for image transforms

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Screen settings
SCREEN_WIDTH = 680
SCREEN_HEIGHT = 680
ACTION_LIM = 512.0 # pushT https://github.com/huggingface/gym-pusht
BACKGROUND_COLOR = (255, 255, 255)

# Colors
ARM_COLOR = (50, 100, 200)
T_COLOR = (200, 50, 50)         # Color for the object
GOAL_T_COLOR = (0, 200, 0)      # Outline color for the goal T

# End-effector parameters
EE_RADIUS = 20.0

# Arm parameters
LINK_LENGTHS = [150.0, 150.0, 150.0, 150.0, 150.0]
INITIAL_ANGLES = [0.29430179, -0.75699092, -1.44649063, -1.0642895, -0.49411484]
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
T = 1000
BATCH_SIZE = 2048
EPOCHS = 600
LEARNING_RATE = 5e-4
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01

# Set the temporal window size.
WINDOW_SIZE = 14

# New constant: dimensionality of image features extracted by VisualEncoder using ResNet18 and spatial softmax
IMAGE_FEATURE_DIM = 32

# Action and condition dimensions
ACTION_DIM = 2
# 4 EE_t-1, EE_t + 32 (image features) = 36
CONDITION_DIM = 4 + IMAGE_FEATURE_DIM

# Temporal Parameters
FPS = 100
SEC_PER_SAMPLE = 0.1
DEMO_DATA_FRAMES = 16
IMG_RES = 96

# Define transform for images (similar to training)
image_transform = transforms.Compose([
    transforms.Resize((IMG_RES, IMG_RES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory for saving training data
TRAINING_DATA_DIR = "training_data"
OUTPUT_DIR = ""

# Diffusion loss masking configuration
DO_MASK_LOSS_FOR_PADDING = True

# -------------------------------------------------------------------
# Dataset and Environment selection for LeRobot integration
# -------------------------------------------------------------------
DATASET_TYPE = "lerobot"      # options: "custom", "lerobot"
LE_ROBOT_GYM_ENV_NAME = "gym_pusht/PushT-v0"

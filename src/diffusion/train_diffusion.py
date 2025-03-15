"""
This module implements the training loop for the diffusion policy model using torch-ema for
Exponential Moving Average, batch-wise learning rate scheduling, and dictionary-based data input.

Key updates:
  - Integrates EMA using the torch-ema package for model stabilization.
  - Removes custom padding and masking logic, assuming normalized, fixed-length inputs.
  - Accepts input data as a dictionary with keys: 'image', 'agent_pos', and 'action'.
  - Uses AdamW hyperparameters (lr=1e-4, weight_decay=1e-6) matching the OTHER code.
  - Steps the learning rate after every batch using a LambdaLR scheduler.
  - Computes total training steps as the product of the number of batches and epochs.
  - Saves model checkpoints and validates periodically.
"""

import os
import math  # For cosine annealing scheduler.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.diffusion.diffusion_policy import DiffusionPolicy
from src.utils.diffusion_utils import get_beta_schedule, compute_alphas
from src.config import *
from einops import rearrange
import wandb
try:
	from kaggle_secrets import UserSecretsClient
except ImportError:
	pass  # Not running on Kaggle
from src.seed import set_seed
from src.datasets.policy_dataset import PolicyDataset
from src.utils.validation_utils import validate_policy

# Import torch-ema package for Exponential Moving Average.
from torch_ema import ExponentialMovingAverage

# Set the random seed for reproducibility.
set_seed(42)

def train():
	"""
	Train the diffusion policy model using torch-ema for EMA and a batch-wise learning rate scheduler.

	This function performs the following steps:
	  - Loads normalized training data in dict format with keys: 'image', 'agent_pos', and 'action'.
	  - Samples diffusion timesteps and applies the forward diffusion process to the action sequences.
	  - Computes the noise prediction loss with optional sqrt weighting.
	  - Optimizes the model using AdamW with the specified hyperparameters.
	  - Updates the learning rate after every batch using a LambdaLR scheduler.
	  - Maintains an EMA of the model parameters using torch-ema.
	  - Logs training metrics to WandB and saves model checkpoints periodically.
	  - Validates the policy at regular intervals.

	Returns:
		None
	"""
	# Retrieve the WandB API key from environment variables.
	use_wandb = False
	secret_label = "WANDB_API_KEY"
	try:
		api_key = UserSecretsClient().get_secret(secret_label)
		if api_key is not None:
			# Log in to WandB using the private API key.
			wandb.login(key=api_key)
			use_wandb = True
	except NameError:
		print("Kaggle secrets not available.")
	except Exception as e:
		print(f"Error initializing WandB: {e}")

	if use_wandb:
		# Initialize a new WandB run with project settings and hyperparameters.
		wandb.init(entity="moribots-personal", project="2d_arm_diffusion", config={
			"epochs": EPOCHS,
			"batch_size": BATCH_SIZE,
			"learning_rate": 1e-4,
			"weight_decay": 1e-6,
			"T": T,
			"window_size": WINDOW_SIZE,
			"dataset": DATASET_TYPE
		})
	else:
		print("Skipping WandB initialization - no API key available.")

	# Enable benchmark mode for optimized performance on fixed-size inputs.
	torch.backends.cudnn.benchmark = True

	# Load normalized data; assume dataset returns a dict with keys: 'image', 'agent_pos', and 'action'.
	dataset = PolicyDataset(
		dataset_type="lerobot",
		data_dir=TRAINING_DATA_DIR,
		pred_horizon=WINDOW_SIZE,
		action_horizon=WINDOW_SIZE // 2,
		obs_horizon=2
	)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
							num_workers=os.cpu_count()-1, pin_memory=True)

	print("len dataloader", len(dataloader))
	print("len dataset", len(dataset))

	# Initialize the diffusion policy model.
	model = DiffusionPolicy(
		action_dim=ACTION_DIM,
		condition_dim=CONDITION_DIM,
		window_size=WINDOW_SIZE
	)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)

	# Enable WandB to automatically log model gradients and parameters.
	if use_wandb:
		wandb.watch(model, log="all")

	# Load pre-trained weights if a checkpoint exists.
	checkpoint_path = os.path.join(OUTPUT_DIR, DATA_SOURCE_DIR, dataset.dataset_type, MODEL_CHECKPOINT_FILENAME)
	if os.path.exists(checkpoint_path):
		print(f"Loading pre-trained policy from {checkpoint_path}")
		state_dict = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(state_dict)

	# Use multiple GPUs if available.
	if torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs for training.")
		model = nn.DataParallel(model)

	# Set up the AdamW optimizer.
	optimizer = optim.AdamW(
		model.parameters(),
		lr=OPTIMIZER_LR,
		betas=OPTIMIZER_BETAS,
		eps=OPTIMIZER_EPS,
		weight_decay=OPTIMIZER_WEIGHT_DECAY
	)

	# Compute total training steps as the product of the number of batches and epochs.
	total_training_steps = len(dataloader) * EPOCHS
	warmup_steps = 1000  # Number of steps for linear warmup.

	# Define a Lambda function for learning rate scheduling with linear warmup and cosine annealing.
	def lr_lambda(current_step):
		if current_step < warmup_steps:
			# Linear warmup: gradually increase lr from 0 to base lr.
			return float(current_step) / float(max(1, warmup_steps))
		# Cosine annealing after warmup.
		progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
		return 0.5 * (1.0 + math.cos(math.pi * progress))

	# Create a scheduler that updates the learning rate every batch.
	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	# Define the MSE loss (without reduction) for element-wise loss computation.
	mse_loss = nn.MSELoss(reduction="none")

	def compute_grad_norm(parameters):
		"""
		Compute the L2 norm of gradients for all parameters.

		Args:
			parameters (iterable): Model parameters.

		Returns:
			float: The total gradient norm.
		"""
		total_norm = 0.0
		parameters = list(filter(lambda p: p.grad is not None, parameters))
		for p in parameters:
			param_norm = p.grad.detach().norm(2)
			total_norm += param_norm.item() ** 2
		return total_norm ** 0.5

	# Generate the beta schedule and compute the cumulative product of alphas.
	betas = get_beta_schedule(T)
	_, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)

	# Initialize the EMA using torch-ema for model stabilization.
	ema = ExponentialMovingAverage(model.parameters(), decay=0.75)

	global_step = 0  # Global step counter for learning rate scheduling.
	# Training loop over epochs.
	for epoch in range(EPOCHS):
		running_loss = 0.0
		running_grad_norm = 0.0
		batch_count = 0

		# Loop over batches.
		for batch in dataloader:
			# Expect batch to be a dict with keys: 'image', 'agent_pos', and 'action'.
			# Transfer input data to the device.
			state = batch['agent_pos'].to(device)
			image = batch['image'].to(device)
			# Assume the action sequences are already normalized and fixed-length.
			action_seq = batch['action'].to(device)
			B = action_seq.size(0)

			# Sample a random diffusion timestep for each sample in the batch.
			t_min = 0
			t_max = T
			t_tensor = torch.randint(t_min, t_max, (B,), device=device)

			# Compute the noise scaling factor from the cumulative product of alphas.
			alpha_bar = rearrange(alphas_cumprod[t_tensor], 'b -> b 1 1')
			noise = torch.randn_like(action_seq)
			# Forward diffusion process: add noise to the clean action sequence.
			x_t = torch.sqrt(alpha_bar) * action_seq + torch.sqrt(1 - alpha_bar) * noise

			# Predict noise using the diffusion policy model.
			noise_pred = model(x_t, t_tensor.float(), state, image)

			# Compute the weight factor based on the noise level.
			weight = torch.sqrt(1 - alpha_bar)

			# Calculate the element-wise MSE loss.
			loss_elements = mse_loss(noise_pred, noise)

			# Optionally weight the loss by sqrt(1 - alpha_bar) if enabled.
			if DO_SQRT_ALPHA_BAR_WEIGHTING:
				loss = torch.mean(weight * loss_elements)
			else:
				loss = torch.mean(loss_elements)

			optimizer.zero_grad()
			loss.backward()

			# Compute gradient norm for logging.
			grad_norm = compute_grad_norm(model.parameters())
			running_grad_norm += grad_norm

			# Apply gradient clipping to prevent exploding gradients.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			optimizer.step()
			# Update EMA weights after the optimizer step using torch-ema.
			ema.update()

			# Update the learning rate scheduler after every batch.
			lr_scheduler.step()
			global_step += 1

			running_loss += loss.item() * B
			batch_count += 1

			# Log batch-level metrics to WandB.
			if use_wandb:
				wandb.log({
					"batch_loss": loss.item(),
					"batch_grad_norm": grad_norm,
					"learning_rate": optimizer.param_groups[0]['lr'],
					"global_step": global_step
				})

		# Compute and log average loss for the epoch.
		avg_loss = running_loss / len(dataset)
		avg_grad_norm = running_grad_norm / batch_count if batch_count > 0 else 0.0

		print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}, Grad Norm: {avg_grad_norm:.6f}")
		if use_wandb:
			wandb.log({
				"epoch": epoch+1,
				"avg_loss": avg_loss,
				"avg_grad_norm": avg_grad_norm
			})

		# Save a checkpoint every 50 epochs.
		if (epoch + 1) % 50 == 0:
			checkpoint_file = os.path.join(OUTPUT_DIR, MODEL_CHECKPOINT_FILENAME)
			torch.save(model.state_dict(), checkpoint_file)
			print(f"Checkpoint saved at epoch {epoch+1}")

		# Validate the policy periodically.
		if (epoch + 1) % VALIDATION_INTERVAL == 0:
			val_reward, _, _ = validate_policy(model, device)  # Validation utility returns reward and additional info.
			print(f"Validation total reward at epoch {epoch+1}: {val_reward}")

	# After training, copy EMA weights to the model for inference.
	ema.copy_to(model.parameters())

	# Save the final model after training completes.
	model_save_path = os.path.join(OUTPUT_DIR, MODEL_CHECKPOINT_FILENAME)
	torch.save(model.state_dict(), model_save_path)
	print(f"Training complete. Model saved as {model_save_path}.")
	
	if use_wandb:
		wandb.finish()

if __name__ == "__main__":
	train()

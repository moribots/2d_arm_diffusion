import os
import json
import csv  # For CSV dumps of training metrics.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization.
from diffusion_policy import DiffusionPolicy
from diffusion_utils import get_beta_schedule, compute_alphas
from config import TRAINING_DATA_DIR, T, BATCH_SIZE, EPOCHS, LEARNING_RATE, ACTION_DIM, CONDITION_DIM
from einops import rearrange

# ------------------------- Dataset Definition -------------------------
class PolicyDataset(Dataset):
	"""
	PolicyDataset loads training samples from JSON files stored in TRAINING_DATA_DIR.
	Each sample is expected to have:
	  - "goal_pose": The desired T pose [x, y, theta].
	  - "T_pose": The current T object pose [x, y, theta].
	  - "action": The end-effector (EE) position (a 2D vector) recorded during data collection.
	The condition is the concatenation of goal_pose and T_pose, forming a 6D vector.
	"""
	def __init__(self, data_dir):
		self.samples = []
		# Loop through all JSON files in the training data directory.
		for filename in os.listdir(data_dir):
			if filename.endswith('.json'):
				filepath = os.path.join(data_dir, filename)
				with open(filepath, 'r') as f:
					data = json.load(f)
					# Add each sample to the dataset.
					self.samples.extend(data)
	
	def __len__(self):
		# Total number of samples.
		return len(self.samples)
	
	def __getitem__(self, idx):
		# Retrieve sample and convert fields to torch tensors.
		sample = self.samples[idx]
		goal_pose = torch.tensor(sample["goal_pose"], dtype=torch.float32)  # Shape: (3,)
		T_pose = torch.tensor(sample["T_pose"], dtype=torch.float32)        # Shape: (3,)
		# Concatenate goal_pose and T_pose to form a 6D conditioning vector.
		condition = torch.cat([goal_pose, T_pose], dim=0)  # Shape: (6,)
		action = torch.tensor(sample["action"], dtype=torch.float32)         # Shape: (2,)
		return condition, action

# ------------------------- Training Loop -------------------------
def train():
	print(f"CUDA is available: {torch.cuda.is_available()}")
	# ------------------------- Visualization and CSV Setup -------------------------
	# Initialize TensorBoard SummaryWriter for logging training metrics.
	# The logs will be saved in the "runs/diffusion_policy_training" directory.
	writer = SummaryWriter(log_dir="runs/diffusion_policy_training")
	# Open a CSV file to dump training metrics.
	# Each row will contain the epoch number and average loss.
	csv_file = open("training_metrics.csv", "w", newline="")
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(["epoch", "avg_loss"])  # Write CSV header.
	# -------------------------------------------------------------------------
	
	# Enable CuDNN benchmark mode for performance optimization when input sizes are constant.
	torch.backends.cudnn.benchmark = True
	
	# Initialize dataset and data loader.
	dataset = PolicyDataset(TRAINING_DATA_DIR)
	dataloader = DataLoader(
		dataset, 
		batch_size=BATCH_SIZE, 
		shuffle=True, 
		num_workers=4,       # Increase this number based on your CPU cores.
		pin_memory=True      # Faster data transfer to GPU.
	)
	
	# Initialize the diffusion policy model.
	model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM)
	
	# Use the AdamW optimizer with the learning rate specified in config.
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
	
	# Mean Squared Error loss will measure the difference between predicted noise and actual noise.
	mse_loss = nn.MSELoss()
	
	# Determine the computing device.
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)
	
	# Generate a linear beta schedule for the diffusion process.
	betas = get_beta_schedule(T)
	# Compute alpha (signal retention factor) and the cumulative product (bar_alpha).
	alphas, alphas_cumprod = compute_alphas(betas)
	alphas_cumprod = alphas_cumprod.to(device)
	
	# Loop over the number of epochs.
	for epoch in range(EPOCHS):
		running_loss = 0.0
		
		# Iterate over mini-batches from the DataLoader.
		for condition, action in dataloader:
			# Move the mini-batch data to the device (GPU or CPU).
			condition = condition.to(device)  # Conditioning signal: 6D vector.
			action = action.to(device)        # Clean action sample: EE position.
			
			# Randomly sample a diffusion timestep 't' for each sample in the batch.
			t = torch.randint(0, T, (action.size(0),), device=device)
			# Extract the cumulative product (bar_alpha) corresponding to t.
			alpha_bar = rearrange(alphas_cumprod[t], 'b -> b 1')  # Shape: (batch, 1)
			
			# Sample noise from a standard normal distribution.
			noise = torch.randn_like(action)
			# Apply the forward diffusion process:
			# x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise,
			# where x_0 is the clean action sample.
			x_t = torch.sqrt(alpha_bar) * action + torch.sqrt(1 - alpha_bar) * noise
			
			# Forward pass through the model to predict the noise.
			# The model is conditioned on the noised sample x_t, the timestep t, and the condition.
			noise_pred = model(x_t, t.float(), condition)
			
			# Compute the mean squared error loss between the predicted noise and the true noise.
			loss = mse_loss(noise_pred, noise)
			
			# Zero the gradients, backpropagate, and update the model parameters.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# Accumulate the batch loss, scaled by the batch size.
			running_loss += loss.item() * action.size(0)
		
		# Compute the average loss for the epoch.
		avg_loss = running_loss / len(dataset)
		print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
		
		# ------------------------- Visualization and CSV Dump -------------------------
		# Log the average loss for this epoch to TensorBoard.
		# This allows you to visualize the training loss over epochs.
		writer.add_scalar("Loss/avg_loss", avg_loss, epoch+1)
		
		# Log histograms for all model weights.
		# This will help you visualize the distribution of weights and monitor their evolution.
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch+1)
		
		# Write the epoch number and average loss to the CSV file.
		csv_writer.writerow([epoch+1, avg_loss])
		# -------------------------------------------------------------------------
		# Overwrite the trained policy every 100 epochs.
		if (epoch + 1) % 100 == 0:
			torch.save(model.state_dict(), "/kaggle/working/diffusion_policy.pth")
			print(f"Checkpoint overwritten at epoch {epoch+1}")
	
	# Save the trained model weights to disk.
	torch.save(model.state_dict(), "diffusion_policy.pth")
	print("Training complete. Model saved as diffusion_policy.pth.")
	
	# Close the TensorBoard writer and CSV file.
	writer.close()
	csv_file.close()

if __name__ == "__main__":
	train()

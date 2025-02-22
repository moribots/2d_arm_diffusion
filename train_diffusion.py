import os
import json
import csv  # For CSV dumps of training metrics.
import math  # Needed for cosine annealing computation.
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
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.samples.extend(data)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        goal_pose = torch.tensor(sample["goal_pose"], dtype=torch.float32)  # (3,)
        T_pose = torch.tensor(sample["T_pose"], dtype=torch.float32)        # (3,)
        condition = torch.cat([goal_pose, T_pose], dim=0)  # (6,)
        action = torch.tensor(sample["action"], dtype=torch.float32)         # (2,)
        return condition, action

# ------------------------- Training Loop -------------------------
def train():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    # ------------------------- Visualization and CSV Setup -------------------------
    writer = SummaryWriter(log_dir="runs/diffusion_policy_training")
    csv_file = open("training_metrics.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "avg_loss"])
    # -------------------------------------------------------------------------
    
    # Enable CuDNN benchmark for optimized kernel selection.
    torch.backends.cudnn.benchmark = True
    
    # Initialize dataset and DataLoader.
    dataset = PolicyDataset(TRAINING_DATA_DIR)
    dataloader = DataLoader(
        dataset, 
        batch_size=256,  # Increased batch size for a P100 GPU.
        shuffle=True, 
        num_workers=4,       # Adjust this number based on your CPU cores.
        pin_memory=True      # Speeds up data transfer to GPU.
    )
    
    # Initialize the diffusion policy model.
    model = DiffusionPolicy(ACTION_DIM, CONDITION_DIM)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss(reduction="none")  # We'll compute elementwise loss to allow weighting.
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # Generate a linear beta schedule for the diffusion process.
    betas = get_beta_schedule(T)
    alphas, alphas_cumprod = compute_alphas(betas)
    alphas_cumprod = alphas_cumprod.to(device)
    
    # ----- Learning Rate Scheduler with Warmup and Cosine Annealing -----
    # For the first 'warmup_epochs', the learning rate increases linearly.
    # Then it decays following a cosine schedule.
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            # Cosine annealing between epoch 'warmup_epochs' and total EPOCHS.
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # ---------------------------------------------------------------------
    
    # Training loop.
    for epoch in range(EPOCHS):
        running_loss = 0.0
        
        for condition, action in dataloader:
            condition = condition.to(device)
            action = action.to(device)
            
            t = torch.randint(0, T, (action.size(0),), device=device)
            alpha_bar = rearrange(alphas_cumprod[t], 'b -> b 1')
            
            noise = torch.randn_like(action)
            x_t = torch.sqrt(alpha_bar) * action + torch.sqrt(1 - alpha_bar) * noise
            
            noise_pred = model(x_t, t.float(), condition)
            
            # --- Loss Weighting by Timestep ---
            # Here we weight the MSE loss by sqrt(1 - alpha_bar) to emphasize difficult timesteps.
            weight = torch.sqrt(1 - alpha_bar)
            loss_elements = mse_loss(noise_pred, noise)
            loss = torch.mean(weight * loss_elements)
            # --- End Loss Weighting ---
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * action.size(0)
        
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
        
        # ----- Logging to TensorBoard and CSV -----
        writer.add_scalar("Loss/avg_loss", avg_loss, epoch+1)
        # Log current learning rate.
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate", current_lr, epoch+1)
        # Log weight histograms.
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch+1)
        # Log gradient norms.
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                writer.add_scalar(f"Gradients/{name}_norm", grad_norm, epoch+1)
                print(f'Gradient norm for {name}: {grad_norm:.6f}')
        csv_writer.writerow([epoch+1, avg_loss])
        # -------------------------------------------
        
        # Step the learning rate scheduler.
        scheduler.step()
        
        # Overwrite the trained policy every 100 epochs.
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), "diffusion_policy.pth")
            print(f"Checkpoint overwritten at epoch {epoch+1}")
    
    torch.save(model.state_dict(), "diffusion_policy.pth")
    print("Training complete. Model saved as diffusion_policy.pth.")
    
    writer.close()
    csv_file.close()

if __name__ == "__main__":
    train()

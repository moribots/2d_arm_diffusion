import os
import parquet
import numpy as np  # Still needed for handling numpy arrays
import torch
from torch.utils.data import Dataset
from einops import rearrange
from PIL import Image
from datasets import load_dataset
from src.utils.normalize import Normalize
from src.config import *

# =============================================================================
# Helper Functions for Sliding Window Extraction and Normalization
# =============================================================================

def create_sample_indices(
		episode_ends: torch.Tensor, sequence_length: int,
		pad_before: int = 0, pad_after: int = 0):
	"""
	Computes sample indices for each episode that allow extraction of fixed-length sequences,
	with optional padding at the beginning and/or end.

	For each episode, possible starting positions (even if they extend outside the episode)
	are considered. Missing data at the boundaries is handled later by repeating the edge value.

	Parameters:
		episode_ends (torch.Tensor): 1D tensor containing the cumulative end indices for episodes.
		sequence_length (int): The total length of the sequence to extract.
		pad_before (int): Number of timesteps allowed for padding at the beginning.
		pad_after (int): Number of timesteps allowed for padding at the end.

	Returns:
		torch.Tensor: A tensor of shape (num_samples, 4) where each row is:
					[buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
	
		 Episode (time steps):  0    1   2  ...  L-1
								  +----+----+-----+ 
		 Desired Sequence: [ pad | available data | pad ]
		 
		 For a window of length L_seq:
		   - Valid starting indices range from -pad_before to (L - L_seq + pad_after)
		   - Negative indices mean missing timesteps at the beginning.
		   - Indices that extend beyond L require padding at the end.
	"""
	indices = []
	for i, end_idx in enumerate(episode_ends):
		# Grab the 0-th or i-1th index as the start. If we grab the 0-th index,
		# we will apply negative padding (pad_before) to the first episode.
		start_idx = 0 if i == 0 else episode_ends[i - 1]
		# Episode length.
		episode_length = end_idx - start_idx

		# Define the valid range for window starting index (allows for padding)
		# min_start can be negative (meaning we need to pad at the beginning)
		min_start = -pad_before
		# max_start can be negative if the episode is shorter than sequence_length.
		# This is to ensure that the last sequence can be extracted even if it extends beyond the episode.
		max_start = episode_length - sequence_length + pad_after

		# Loop over [min_start, max_start] inclusive to get all possible starting indices.
		# This will include negative indices for missing timesteps at the beginning.
		for idx in range(min_start, max_start + 1):
			# Compute the actual indices in the raw data (adjust for episode offset)
			# If idx is negative, we use 0 as the starting point for available data.
			buffer_start_idx = max(idx, 0) + start_idx
			# Similarly, if idx+sequence_length goes beyond the episode, limit it to episode_length.
			buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx

			# Determine how many timesteps are missing (and thus need padding)
			# If idx is negative, it means we are missing data at the beginning.
			start_offset = max(0, -idx)
			# If idx + sequence_length exceeds the episode length, we need to pad at the end.
			end_offset = max(0, (idx + sequence_length) - episode_length)

			# Determine where in the fixed-size sample the available data should be inserted.
			# The data should start at index start_offset and end at sequence_length - end_offset.
			sample_start_idx = start_offset
			sample_end_idx = sequence_length - end_offset
			# Append computed indices.
			indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
	return torch.tensor(indices, dtype=torch.long)


def sample_sequence(train_data, sequence_length,
					buffer_start_idx, buffer_end_idx,
					sample_start_idx, sample_end_idx):
	"""
	Extracts a subsequence from the provided training data using buffer indices and
	applies padding by repeating edge data to ensure the returned sequence has the desired fixed length.

	Parameters:
		train_data (dict): Dictionary mapping keys (e.g., 'image', 'agent_pos', 'action') to torch tensors.
		sequence_length (int): The fixed length of the desired sequence.
		buffer_start_idx (int): Start index in the original data array.
		buffer_end_idx (int): End index in the original data array.
		sample_start_idx (int): Start index in the output sequence where actual data should be inserted.
		sample_end_idx (int): End index in the output sequence where actual data should be inserted.

	Returns:
		dict: A dictionary with the same keys as train_data, where each value is a torch tensor
			  of shape (sequence_length, ...) that includes the extracted sequence with proper padding.

	Explanation:
		- For each key in train_data, the function extracts the available slice [buffer_start_idx:buffer_end_idx].
		- If the extracted slice does not fill the entire sequence_length (i.e. if padding is needed),
		  a new tensor of zeros is created.
			- The missing values at the beginning (if sample_start_idx > 0) are filled with the first
			  available value from the slice.
			- The missing values at the end (if sample_end_idx < sequence_length) are filled with the last
			  available value from the slice.
			- The actual data is inserted into the middle of the new array at positions [sample_start_idx:sample_end_idx].
	"""
	result = {}
	for key, input_arr in train_data.items():
		# Extract the contiguous available data slice.
		sample = input_arr[buffer_start_idx:buffer_end_idx]
		data = sample  # Default, if no padding is needed.

		# Check if extremity padding is required.
		if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
			# Set up the full desired shape.
			data = torch.zeros(
				size=(sequence_length,) + tuple(input_arr.shape[1:]),
				dtype=input_arr.dtype,
				device=input_arr.device)
			# Pad beginning by repeating the first available value.
			if sample_start_idx > 0:
				data[:sample_start_idx] = sample[0]
			# Pad the end by repeating the last available value.
			if sample_end_idx < sequence_length:
				data[sample_end_idx:] = sample[-1]
			# Place the available data into the container using the computed indices.
			data[sample_start_idx:sample_end_idx] = sample
		result[key] = data
	return result

# =============================================================================
# PolicyDataset Using a Sliding Window Approach
# =============================================================================

class PolicyDataset(Dataset):
	"""
	PolicyDataset loads training samples for the diffusion policy using a sliding window approach.
	It supports two types of inputs:
	
	  - "lerobot": Loads from the Hugging Face dataset "lerobot/pusht_image".
	  - "custom": Loads from parquet files in a specified directory, with associated image files.
	
	The dataset constructs contiguous sequences from episodic data. For each sample, it returns:
	  - 'image': A sequence of images (shape: (obs_horizon, 3, 96, 96))
	  - 'agent_pos': A sequence of agent positions (shape: (obs_horizon, 2))
	  - 'action': A sequence of actions (shape: (pred_horizon, 2))
	
	The full sequence (of length pred_horizon) is extracted using a sliding window with padding,
	and then the observation portion (first obs_horizon steps) is used for images and agent positions.
	"""
	def __init__(self, dataset_type: str, data_dir: str,
				 pred_horizon: int, obs_horizon: int, action_horizon: int):
		"""
		Initializes the PolicyDataset.
		
		Steps:
		  1. Loads and groups samples by episode. For "lerobot", it uses the Hugging Face dataset;
			 for "custom", it reads parquet files from data_dir and loads images from disk.
		  2. Sorts each episode by 'frame_index' and accumulates data into contiguous arrays:
			 - images: (N, 3, 96, 96)
			 - agent_pos: (N, 2) extracted from 'observation.state'
			 - action: (N, 2)
		  3. Computes episode boundaries (episode_ends) based on the accumulated data.
		  4. Computes sliding window sample indices using create_sample_indices().
		  5. Computes per-key normalization statistics and normalizes agent_pos and action data to [-1,1].
			 and images to [0, 1].
		  6. Saves the normalization stats and preprocessed data for use in __getitem__, where a custom
		  	 schedule of observations and actions is returned (e.g 2 obs with 16 actions).
			 
		Parameters:
			dataset_type (str): Either "lerobot" or "custom" to indicate the data source.
			data_dir (str): Directory where the custom parquet (and image) files reside (or can be ignored for lerobot).
			pred_horizon (int): Total length of the sequence to extract.
			obs_horizon (int): Number of timesteps to use as the observation (from the beginning of the sequence).
			action_horizon (int): Horizon for action data; used for padding at the end.
		"""
		 # Store dataset_type as an instance attribute
		self.dataset_type = dataset_type
		
		# Normalizer.
		self.normalize = Normalize.compute_from_limits()
		# Lists to accumulate contiguous data across all episodes.
		images_list = []
		agent_pos_list = []
		actions_list = []
		episode_ends = []  # Will hold cumulative counts of samples per episode.
		current_count = 0  # Running count of samples.

		episodes = {}

		# Load and group episodes based on dataset_type.
		if dataset_type == "lerobot":
			# Load Hugging Face dataset.
			hf_dataset = load_dataset("lerobot/pusht_image", split="train")
			print(f'Loaded {len(hf_dataset)} samples from LeRobot dataset.')
			for sample in hf_dataset:
				ep = sample["episode_index"]
				episodes.setdefault(ep, []).append(sample)
		else:
			# Custom data: load all parquet files from data_dir.
			for filename in os.listdir(data_dir):
				if filename.endswith('.pqt'):
					filepath = os.path.join(data_dir, filename)
					with open(filepath, 'rb') as f:
						data = parquet.load(f)
						for sample in data:
							ep = sample["episode_index"]
							episodes.setdefault(ep, []).append(sample)
			print(f'Loaded {len(episodes)} episodes from custom dataset.')

		# Process each episode.
		for ep, ep_samples in episodes.items():
			# Sort samples in the episode by frame_index.
			ep_samples = sorted(ep_samples, key=lambda s: s["frame_index"])
			# Skip episodes that are too short for the sliding window.
			if len(ep_samples) < pred_horizon:
				continue

			# Process each sample (assumed to be one timestep).
			for sample in ep_samples:
				# --------------------------
				# Process image data.
				# --------------------------
				if dataset_type == "lerobot":
					# For lerobot, image data is assumed to be a NumPy array.
					img = sample["observation.image"]
					
					# Convert to PIL Image if it's a numpy array
					if isinstance(img, np.ndarray):
						img = Image.fromarray(img)
					
					# Apply the image_transform from config - this handles normalization
					# and conversion to tensor in the proper format
					img_tensor = image_transform(img)
					
					# In case image_transform doesn't handle reshaping, keep the shape check
					if img_tensor.shape[-1] == 3 and img_tensor.ndim == 3:  # Still in HWC format
						img_tensor = rearrange(img_tensor, 'h w c -> c h w')
				else:
					# For custom, assume 'observation.image' is a filename.
					# Construct the full path to the image file.
					img_filename = sample["observation.image"]
					img_path = os.path.join(data_dir, "images", img_filename)
					if not os.path.exists(img_path):
						raise FileNotFoundError(f"Image file not found: {img_path}")
					
					# Open the image and convert to RGB.
					img_pil = Image.open(img_path).convert("RGB")
					
					# Apply the image_transform from config
					img_tensor = image_transform(img_pil)
					
					# Keep shape check in case image_transform doesn't handle reshaping
					if img_tensor.shape[-1] == 3 and img_tensor.ndim == 3:  # Still in HWC format
						img_tensor = rearrange(img_tensor, 'h w c -> c h w')

				images_list.append(img_tensor)

				# --------------------------
				# Process agent position (state) data.
				# --------------------------
				# TODO(mrahme): add goal conditioning.
				state = sample["observation.state"]
				agent_pos = torch.tensor(state)
				agent_pos_list.append(agent_pos)

				# --------------------------
				# Process action data.
				# --------------------------
				action = torch.tensor(sample["action"])
				actions_list.append(action)

			# Update episode boundary.
			current_count += len(ep_samples)
			episode_ends.append(current_count)

		# Convert lists to contiguous tensors
		# Images: shape (N, 3, 96, 96) -- assume all images are 96x96.
		train_image_data = torch.stack(images_list, dim=0)
		# agent_pos: shape (N, 2)
		train_agent_pos = torch.stack(agent_pos_list, dim=0)
		# action: shape (N, 2)
		train_action = torch.stack(actions_list, dim=0)
		episode_ends = torch.tensor(episode_ends)

		# ------------------------------
		# Compute sliding window sample indices.
		# ------------------------------
		# Using the helper function to get valid sequence indices for each episode.
		# Pad before: obs_horizon - 1 (to include enough context before prediction horizon)
		# Pad after: action_horizon - 1 (to include enough context after the observed sequence)
		self.indices = create_sample_indices(
			episode_ends=episode_ends,
			sequence_length=pred_horizon,
			pad_before=obs_horizon - 1,
			pad_after=action_horizon - 1
		)

		# ------------------------------
		# Normalize agent_pos and action data.
		# ------------------------------
		normalized_train_data = {}
		# Normalize agent positions using the Normalize class
		normalized_train_data['agent_pos'] = self.normalize.normalize_condition(train_agent_pos)
		# Normalize actions using the Normalize class
		normalized_train_data['action'] = self.normalize.normalize_action(train_action)
		# Images are normalized in an earlier step.
		normalized_train_data['image'] = train_image_data

		# Save attributes for use in __getitem__.
		self.normalized_train_data = normalized_train_data
		self.pred_horizon = pred_horizon
		self.obs_horizon = obs_horizon
		self.action_horizon = action_horizon

	def __len__(self):
		"""
		Returns the total number of sliding window samples available.
		"""
		return len(self.indices)

	def __getitem__(self, idx):
		"""
		Retrieves a sample from the dataset.
		
		For the given index, the function:
		  1. Retrieves the precomputed indices for slicing the data.
		  2. Extracts the fixed-length sequence for each modality using sample_sequence().
		  3. Truncates the 'image' and 'agent_pos' arrays to the observation horizon.
		  
		Returns:
			dict: A dictionary with keys:
				  - 'image': shape (obs_horizon, 3, 96, 96)
				  - 'agent_pos': shape (obs_horizon*2,) - flattened for model input
				  - 'action': shape (pred_horizon, 2)
		"""
		# Unpack the four indices for the current sample.
		buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

		# Extract the sliding window sequence with proper padding.
		nsample = sample_sequence(
			train_data=self.normalized_train_data,
			sequence_length=self.pred_horizon,
			buffer_start_idx=buffer_start_idx,
			buffer_end_idx=buffer_end_idx,
			sample_start_idx=sample_start_idx,
			sample_end_idx=sample_end_idx
		)

		# For observations, only return the first obs_horizon timesteps.
		nsample['image'] = nsample['image'][:self.obs_horizon]
		# Store the original 2D agent positions
		orig_agent_pos = nsample['agent_pos'][:self.obs_horizon]
		# Flatten agent positions using einops rearrange instead of reshape
		nsample['agent_pos'] = rearrange(orig_agent_pos, 'h d -> (h d)')
		# 'action' remains as the full pred_horizon sequence.
		return nsample

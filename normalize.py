import torch
import json

class Normalize:
	def __init__(self, condition_mean, condition_std, action_mean, action_std):
		self.condition_mean = condition_mean  # Tensor of shape (condition_dim,)
		self.condition_std = condition_std    # Tensor of shape (condition_dim,)
		self.action_mean = action_mean        # Tensor of shape (action_dim,)
		self.action_std = action_std          # Tensor of shape (action_dim,)

	@classmethod
	def compute_from_samples(cls, samples):
		"""
		Compute normalization statistics from a list of samples.
		Each sample is expected to be a dict with keys "goal_pose", "T_pose", and "action".
		"""
		conditions = []
		actions_list = []
		for sample in samples:
			# Process condition.
			goal_pose = torch.tensor(sample["goal_pose"], dtype=torch.float32)
			if isinstance(sample["T_pose"][0], (list, tuple)):
				T_pose_seq = torch.tensor(sample["T_pose"], dtype=torch.float32)
				if T_pose_seq.shape[0] >= 2:
					T_pose_prev = T_pose_seq[-2]
					T_pose_curr = T_pose_seq[-1]
				else:
					T_pose_prev = T_pose_seq[-1]
					T_pose_curr = T_pose_seq[-1]
			else:
				T_pose = torch.tensor(sample["T_pose"], dtype=torch.float32)
				T_pose_prev = T_pose
				T_pose_curr = T_pose
			cond = torch.cat([goal_pose, T_pose_prev, T_pose_curr], dim=0)
			conditions.append(cond)
			
			# Process action.
			action_data = sample["action"]
			if isinstance(action_data[0], (list, tuple)):
				act = torch.tensor(action_data, dtype=torch.float32)
			else:
				act = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
			actions_list.append(act)
		conditions_cat = torch.stack(conditions, dim=0)  # (num_samples, condition_dim)
		condition_mean = conditions_cat.mean(dim=0)
		condition_std = conditions_cat.std(dim=0) + 1e-6  # avoid division by zero

		actions_cat = torch.cat(actions_list, dim=0)  # (total_timesteps, action_dim)
		action_mean = actions_cat.mean(dim=0)
		action_std = actions_cat.std(dim=0) + 1e-6

		return cls(condition_mean, condition_std, action_mean, action_std)

	def normalize_condition(self, condition):
		"""
		Normalize a condition tensor.
		"""
		return (condition - self.condition_mean) / self.condition_std

	def normalize_action(self, action):
		"""
		Normalize an action tensor.
		"""
		return (action - self.action_mean) / self.action_std

	def unnormalize_action(self, action):
		"""
		Un-normalize an action tensor.
		"""
		return action * self.action_std + self.action_mean

	def to_json(self):
		return {
			"condition_mean": self.condition_mean.tolist(),
			"condition_std": self.condition_std.tolist(),
			"action_mean": self.action_mean.tolist(),
			"action_std": self.action_std.tolist()
		}

	def save(self, filepath):
		with open(filepath, "w") as f:
			json.dump(self.to_json(), f)

	@classmethod
	def load(cls, filepath, device=None):
		with open(filepath, "r") as f:
			data = json.load(f)
		condition_mean = torch.tensor(data["condition_mean"], dtype=torch.float32, device=device)
		condition_std = torch.tensor(data["condition_std"], dtype=torch.float32, device=device)
		action_mean = torch.tensor(data["action_mean"], dtype=torch.float32, device=device)
		action_std = torch.tensor(data["action_std"], dtype=torch.float32, device=device)
		return cls(condition_mean, condition_std, action_mean, action_std)

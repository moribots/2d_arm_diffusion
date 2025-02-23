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
        Each sample is expected to have:
          - "observation": a dict with
              - "state": a list of two states (each a 2D vector), which will be flattened (4 numbers)
          - "action": a list of actions (each a 2D vector) or a single action.
        """
        conditions = []
        actions_list = []
        for sample in samples:
            # Process condition from observation state.
            if "observation" not in sample or "state" not in sample["observation"]:
                raise KeyError("Sample must contain observation['state']")
            state = torch.tensor(sample["observation"]["state"], dtype=torch.float32)  # shape (2,2)
            cond = state.flatten()  # shape (4,)
            conditions.append(cond)
            
            # Process action.
            if "action" not in sample:
                raise KeyError("Sample does not contain 'action'")
            action_data = sample["action"]
            if isinstance(action_data[0], (list, tuple)):
                act = torch.tensor(action_data, dtype=torch.float32)
            else:
                act = torch.tensor(action_data, dtype=torch.float32).unsqueeze(0)
            actions_list.append(act)
        conditions_cat = torch.stack(conditions, dim=0)  # (num_samples, 4)
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

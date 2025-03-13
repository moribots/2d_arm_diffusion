"""
Utility functions used throughout the project.
"""

import math
import random
import torch
import pygame
from src.config import *
import os

def random_t_pose():
	"""
	Generate a random pose [x, y, theta] within screen bounds.
	
	Returns:
		Tensor: Pose as (x, y, theta).
	"""
	margin = 100
	x = random.uniform(margin, SCREEN_WIDTH - margin)
	y = random.uniform(margin, SCREEN_HEIGHT - margin)
	theta = random.uniform(-math.pi, math.pi)
	return torch.tensor([x, y, theta], dtype=torch.float32)

def angle_diff(a: float, b: float) -> float:
	"""
	Compute the wrapped difference between two angles (in radians).
	
	Returns a value in [-pi, pi].
	
	Args:
		a (float): Angle a.
		b (float): Angle b.
		
	Returns:
		float: Wrapped angle difference.
	"""
	return (a - b + math.pi) % (2 * math.pi) - math.pi

def draw_arrow(surface, color, start, end, width=3, head_length=10, head_angle=30):
	"""
	Draw an arrow on the given pygame surface from start to end.
	
	Args:
		surface: Pygame surface.
		color: Color tuple.
		start: Starting coordinate (x, y).
		end: Ending coordinate (x, y).
		width (int): Width of the arrow.
		head_length (int): Length of the arrow head.
		head_angle (int): Angle of the arrow head.
	"""
	pygame.draw.line(surface, color, start, end, width)
	dx = end[0] - start[0]
	dy = end[1] - start[1]
	angle = math.atan2(dy, dx)
	left_angle = angle + math.radians(head_angle)
	right_angle = angle - math.radians(head_angle)
	left_end = (end[0] - head_length * math.cos(left_angle), end[1] - head_length * math.sin(left_angle))
	right_end = (end[0] - head_length * math.cos(right_angle), end[1] - head_length * math.sin(right_angle))
	pygame.draw.line(surface, color, end, left_end, width)
	pygame.draw.line(surface, color, end, right_end, width)

def get_training_data_dir(env_type: str) -> str:
	"""
	Get the training data directory for a given environment type.
	
	Args:
		env_type (str): "lerobot" or "custom".
		
	Returns:
		str: Directory path.
	"""
	if env_type == "lerobot":
		return "lerobot/" + TRAINING_DATA_DIR
	else:
		return "custom/" + TRAINING_DATA_DIR

def recompute_normalization_stats(env_type: str, norm_stats_path: str):
	"""
	Recompute normalization statistics from the dataset.
	
	For LeRobot, loads the Hugging Face dataset.
	For custom data, reads JSON files in the training data directory.
	
	Args:
		env_type (str): Environment type.
		norm_stats_path (str): Path to save normalization statistics.
	"""
	print(f'Normalization statistics path: {norm_stats_path}')
	if env_type == "lerobot":
		from datasets import load_dataset
		dataset = load_dataset("lerobot/pusht")
		if dataset is not None:
			print("Loading training data from LeRobot dataset...")
			training_samples = list(dataset["train"])
			print("Recomputing normalization statistics...")
			new_norm = __import__("normalize").Normalize.compute_from_limits()
			new_norm.save(norm_stats_path + "normalization_stats.parquet")
		else:
			print("Dataset not available; using existing normalization stats.")
	else:
		training_samples = []
		for filename in os.listdir(get_training_data_dir("custom")):
			if filename.endswith('.json'):
				filepath = os.path.join(get_training_data_dir("custom"), filename)
				with open(filepath, "r") as f:
					data = json.load(f)
					training_samples.extend(data)
		if training_samples:
			print("Recomputing normalization statistics from custom training data...")
			new_norm = __import__("normalize").Normalize.compute_from_limits()
			new_norm.save(norm_stats_path + "normalization_stats.parquet")
			return new_norm
		else:
			print("No custom training samples found; using existing normalization stats.")
		return None

def compute_ee_velocity(ee_pos: torch.Tensor, prev_ee_pos: torch.Tensor, dt: float) -> torch.Tensor:
	"""
	Compute the end-effector velocity based on current and previous positions.
	
	Args:
		ee_pos (Tensor): Current end-effector position.
		prev_ee_pos (Tensor): Previous end-effector position.
		dt (float): Time difference.
		
	Returns:
		Tensor: Computed velocity.
	"""
	if prev_ee_pos is None:
		return torch.zeros_like(ee_pos)
	return (ee_pos - prev_ee_pos) / dt

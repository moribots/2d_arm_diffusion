import math
import random
import torch
import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT

def random_t_pose():
	"""
	Generate a random pose [x, y, theta] within screen bounds.
	A margin is applied to ensure the object stays fully visible.
	"""
	margin = 100
	x = random.uniform(margin, SCREEN_WIDTH - margin)
	y = random.uniform(margin, SCREEN_HEIGHT - margin)
	theta = random.uniform(-math.pi, math.pi)
	return torch.tensor([x, y, theta], dtype=torch.float32)

def angle_diff(a: float, b: float) -> float:
	"""
	Compute the wrapped difference between two angles (radians).
	Returns a value in [-pi, pi].
	"""
	return (a - b + math.pi) % (2 * math.pi) - math.pi

def draw_arrow(surface, color, start, end, width=3, head_length=10, head_angle=30):
	"""
	Draw an arrow on the given surface from start to end.
	"""
	pygame.draw.line(surface, color, start, end, width)
	dx = end[0] - start[0]
	dy = end[1] - start[1]
	angle = math.atan2(dy, dx)
	left_angle = angle + math.radians(head_angle)
	right_angle = angle - math.radians(head_angle)
	left_end = (end[0] - head_length * math.cos(left_angle),
				end[1] - head_length * math.sin(left_angle))
	right_end = (end[0] - head_length * math.cos(right_angle),
				 end[1] - head_length * math.sin(right_angle))
	pygame.draw.line(surface, color, end, left_end, width)
	pygame.draw.line(surface, color, end, right_end, width)

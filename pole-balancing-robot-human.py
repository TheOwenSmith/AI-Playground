import gymnasium as gym
import pygame
import numpy as np
import random

env = gym.make("CartPole-v1", render_mode="human")
env.unwrapped.theta_threshold_radians = np.pi / 2
obs, info = env.reset()

pygame.init()

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action = 0 if random.random() < 0.5 else 1
    if keys[pygame.K_LEFT]:
        action = 0  # push left
    elif keys[pygame.K_RIGHT]:
        action = 1  # push right

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    clock.tick(30)  # 10 fps, slowed down so it's playable

env.close()
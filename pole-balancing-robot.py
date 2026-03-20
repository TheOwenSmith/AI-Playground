import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

# Match pole-balancing-actual.py: linspace + np.digitize → 11 bins per dim (indices 0..10)
pos_space = np.linspace(-2.4, 2.4, 10)
vel_space = np.linspace(-4, 4, 10)
ang_space = np.linspace(-12 * (np.pi / 180), 12 * (np.pi / 180), 10)
ang_vel_space = np.linspace(-4, 4, 10)

gamma = 0.99
alpha = 0.1

CACHE_FILE = "pole_balancing_cache.npz"
Q_SHAPE = (len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1, len(ang_vel_space) + 1, 2)


try:
    cache = np.load(CACHE_FILE)
    Q = cache["Q"].copy()
    N = cache["N"].copy() if "N" in cache else np.zeros(Q_SHAPE, dtype=np.int32)
    if Q.shape != Q_SHAPE:
        Q = np.zeros(Q_SHAPE)
        N = np.zeros(Q_SHAPE, dtype=np.int32)
        print(f"Cache shape mismatch; reinitializing Q, N with shape {Q_SHAPE}")
    else:
        if N.shape != Q_SHAPE:
            N = np.zeros(Q_SHAPE, dtype=np.int32)
        print(f"Loaded Q and N from {CACHE_FILE}")
except FileNotFoundError:
    Q = np.zeros(Q_SHAPE)
    N = np.zeros(Q_SHAPE, dtype=np.int32)

def discretize_state(state):
    state_p = np.digitize(state[0], pos_space)
    state_v = np.digitize(state[1], vel_space)
    state_a = np.digitize(state[2], ang_space)
    state_av = np.digitize(state[3], ang_vel_space)
    return (state_p, state_v, state_a, state_av)

def main(train = True):
    print("Training started. Press Ctrl+C in the terminal to quit and save.")
    episode = 1
    max_reward = 0
    episode_rewards = []
    epsilon = 1.0 if train else 0
    epsilon_decay_rate = 0.00005

    mean_rewards = []
    # Train until Ctrl+C; in eval mode run until Ctrl+C. (Don't stop just because epsilon hit 0.)
    while True:
        init_state, info = env.reset()
        current_state = discretize_state(init_state)

        prev_state = None
        reward_current_episode = 0

        running = True
        while running:
            # compute action
            action = take_action(current_state, epsilon)

            # take action
            prev_state = current_state
            obs, reward, terminated, truncated, info = env.step(action)
            reward_current_episode += reward

            # discretize new state
            current_state = discretize_state(obs)

            # update Q
            if train:
                Q[prev_state][action] += alpha * (
                    reward + gamma * np.max(Q[current_state]) - Q[prev_state][action]
                )
            
            if terminated or truncated:
                if truncated and not train:
                    print('AI Won!')
                running = False

        max_reward = max(max_reward, reward_current_episode)
        episode_rewards.append(reward_current_episode)

        if train:
            epsilon = max(epsilon - epsilon_decay_rate, 0)
        episode += 1

        if episode % 100 == 0:
            mean_last_100 = np.mean(episode_rewards)
            mean_rewards.append(mean_last_100)
            episode_rewards = []
            print(f'Episode: {episode} Epsilon: {epsilon} Max Reward: {max_reward} Mean last 100 rewards: {mean_last_100:.1f}')

def take_action(state, epsilon):
    # epsilon-greedey
    if (random.random() < epsilon):
        return random.choice([0, 1])
    else:
        return np.argmax(Q[state])
    
    # ucb
    # state_visits = np.sum(N[state])
    # ucb_values = np.where(N[state] == 0, np.inf, Q[state] + c * np.sqrt(np.log(state_visits) / N[state]))
    # action = np.argmax(ucb_values)
    # N[state][action] += 1
    # return action

def save_cache():
    np.savez(CACHE_FILE, Q=Q, N=N)
    print(f"Cached Q and N to {CACHE_FILE}")

if __name__ == '__main__':
    train = '--train' in sys.argv
    if train:
        env = gym.make("CartPole-v1")
    else:
        env = gym.make("CartPole-v1", render_mode="human")
        env = env.unwrapped

    try:
        main(train = train)
    except KeyboardInterrupt:
        env.close()
        if train:
            save_cache()
        print('Stopped by user (Ctrl+C).')

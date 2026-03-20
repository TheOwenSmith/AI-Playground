import sys
import gymnasium as gym
import random
import numpy as np

CACHE_FILE = 'ramp_cache.npz'
epsilon_decay = 0.00005
alpha = 0.1
gamma = 0.99

pos_space = np.linspace(-1.2, 0.6, 20)
vel_space = np.linspace(-0.07, 0.07, 20)

Q_SHAPE = (len(pos_space) + 1, len(vel_space) + 1, 3)
N = np.zeros(Q_SHAPE, dtype=np.int32)
Q = np.random.uniform(0, 1, size=Q_SHAPE)

def obs_to_state(obs):
    pos, vel = obs
    pos_discrete = np.digitize(pos, pos_space)
    vel_discrete = np.digitize(vel, vel_space)
    return (pos_discrete, vel_discrete)

t = 0
def decide_action(state, c):
    # epsilon greedy
    # if random.random() < epsilon:
    #     return random.choice([0, 1, 2])
    # else:
    #     return np.argmax(Q[state])

    # upper confidence bound
    global t
    t += 1
    action = np.argmax(Q[state] + np.sqrt(np.log(t) / N[state]))
    N[state][action] += 1
    return action

def main(train = True):
    env = gym.make("MountainCar-v0", render_mode="human" if not train else None)
    if not train:
        env = env.unwrapped
    
    episode = 1
    # epsilon = 1 if train else 0
    c = 1 if train else 0
    episode_rewards = []
    
    while True:
        obs, _info = env.reset()
        state = obs_to_state(obs)

        total_reward = 0

        running = True
        while running:
            # determine action
            # action = decide_action(state, epsilon)
            action = decide_action(state, c)

            # take action
            obs, reward, terminated, truncated, info = env.step(action)

            # update Q
            prev_state = state
            state = obs_to_state(obs)
            Q[prev_state][action] += alpha * (reward + gamma * np.max(Q[state]) - Q[prev_state][action])

            # update constants
            total_reward += reward
            if terminated or truncated:
                episode_rewards.append(total_reward)
                running = False

        if episode % 100 == 0:
            mean_last_100 = np.mean(episode_rewards)
            episode_rewards = []
            # print(f'Episode: {episode} Epsilon: {epsilon} Mean last 100 rewards: {mean_last_100:.1f}')
            print(f'Episode: {episode} Mean last 100 rewards: {mean_last_100:.1f}')
        
        # epsilon = max(0, epsilon - epsilon_decay)
        episode += 1

if __name__ == "__main__":
    if input('Do you want to load Q from cache (y or n): ') == 'y':
        try:
            cache = np.load(CACHE_FILE)
            if 'Q' in cache:
                Q = cache['Q']
                print('Loaded Q from cache')
        except FileNotFoundError:
            print('Q could not be loaded from cache')
    
    train = '--train' in sys.argv
    try:
        main(train = train)
    except KeyboardInterrupt:
        if train and input('Do you want to cache the Q algorithm (y or n):') == 'y':
            np.savez(CACHE_FILE, Q=Q)
            print(f"Cached Q to {CACHE_FILE}")

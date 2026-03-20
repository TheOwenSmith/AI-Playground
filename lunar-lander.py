import gymnasium as gym
import sys
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# hyperparameters
REPLAY_BUFFER_SIZE = 50_000
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
TARGET_SYNC_EVERY = 500

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999995
gamma = 0.99

CACHE_FILE = 'lunar-lander-dqn.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pygame.init()
clock = pygame.time.Clock()

class DQN(nn.Module):
    def __init__(self, input, output):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Intialize DQN
Q = DQN(8, 4).to(device)
Q_target = DQN(8, 4).to(device)
Q_target.load_state_dict(Q.state_dict())
optimizer = optim.Adam(Q.parameters(), lr=LEARNING_RATE)

def decide_action(state, epsilon):
    with torch.no_grad():
        q_values = Q(state.unsqueeze(0))
    
    if random.random() < epsilon:
        action = random.choice([0, 1, 2, 3])
    else:
        action = q_values.argmax(dim=1).item()
    
    return action, q_values[0, action].item()

def train_step(batch):
    states, actions, rewards, new_states, is_terminals = zip(*batch)

    # convert batch to tensors
    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    new_states = torch.stack(new_states).to(device)
    is_terminals = torch.tensor(is_terminals, dtype=torch.bool).to(device)

    # compute targets and actual
    with torch.no_grad():
        max_next_q = Q_target(new_states).max(dim=1).values
        targets = torch.where(is_terminals, rewards, rewards + gamma * max_next_q)
    
    actual = Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # gradient descent
    loss = nn.MSELoss()(actual, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(Q.parameters(), 100)
    optimizer.step()

    return loss.detach().item()

def main(train, play):
    if train and play:
        raise Exception('Cannot play and train at the same time')
    
    env = gym.make("LunarLander-v3", render_mode='human' if not train else None)

    # play
    if play:
        env.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            action = 0
            if keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]:
                action = 2 # main engine
            elif keys[pygame.K_LEFT]:
                action = 3 # right engine
            elif keys[pygame.K_RIGHT]:
                action = 1 # left engine

            _obs, _reward, terminated, truncated, _info = env.step(action)

            if terminated or truncated:
                if terminated:
                    print('LOSE')
                else:
                    print('Win')
                env.reset()

            clock.tick(30)

    # train / observe
    epsilon = epsilon_start if train else 0
    D = deque([], maxlen=REPLAY_BUFFER_SIZE)
    episode = 0
    train_steps = 0

    # constants
    average_q_value_history = []
    average_episode_reward_history = []
    average_loss_history = []
    epsilon_history = []

    steps_since_update_user = 0
    train_steps_since_update_user = 0
    average_q_value = 0
    average_episode_reward = 0
    average_loss = 0

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DQN Training")
    
    while True:
        episode += 1
        total_reward = 0

        obs, _info = env.reset()
        state = torch.tensor(obs, dtype=torch.float32)

        while True:
            steps_since_update_user += 1

            # make decision
            action, q_value = decide_action(state, epsilon)
            average_q_value += (q_value - average_q_value) / steps_since_update_user

            # execute decision
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            new_state = torch.tensor(obs, dtype=torch.float32)
            is_terminal = terminated or truncated

            # update replay buffer
            transition = (state, action, reward, new_state, is_terminal)
            D.append(transition)
            state = new_state
            
            # update DQN
            if train and len(D) == REPLAY_BUFFER_SIZE:
                train_steps += 1
                train_steps_since_update_user += 1

                transitions = random.sample(D, BATCH_SIZE)
                loss = train_step(transitions)
                average_loss += (loss - average_loss) / train_steps_since_update_user
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                if train_steps % TARGET_SYNC_EVERY == 0:
                    Q_target.load_state_dict(Q.state_dict())
            
            if is_terminal:
                break

        average_episode_reward += total_reward / 20
        if episode % 20 == 0:
            # update history
            if train_steps_since_update_user > 0:
                average_q_value_history.append(average_q_value)
                average_episode_reward_history.append(average_episode_reward)
                average_loss_history.append(average_loss)
                epsilon_history.append(epsilon)

            # log results
            loss_str = 'n/a' if train_steps_since_update_user == 0 else f'{average_loss:.5f}'
            print(f'Episode {episode} | Avg Q Value: {average_q_value:.2f} | Avg Episode Reward: {average_episode_reward:.2f} | Loss: {loss_str} | Epsilon: {epsilon:.3f}')

            # plot results
            if train_steps_since_update_user > 0:
                axes[0,0].cla()
                axes[0,0].plot(average_q_value_history)
                axes[0,0].set_title("Avg Q Value")
                axes[0,0].set_xlabel("20 Episodes")

                axes[0,1].cla()
                axes[0,1].plot(average_episode_reward_history)
                axes[0,1].set_title("Avg Episode Reward")
                axes[0,1].set_xlabel("20 Episodes")

                axes[1,0].cla()
                axes[1,0].plot(average_loss_history)
                axes[1,0].set_title("Loss")
                axes[1,0].set_xlabel("20 Episodes")

                axes[1,1].cla()
                axes[1,1].plot(epsilon_history)
                axes[1,1].set_title("Epsilon")
                axes[1,1].set_xlabel("20 Episodes")

            plt.tight_layout()
            plt.pause(0.001)

            # reset constants
            steps_since_update_user = 0
            train_steps_since_update_user = 0
            average_q_value = 0
            average_episode_reward = 0
            average_loss = 0

if __name__ == '__main__':
    train = '--train' in sys.argv
    play = '--play' in sys.argv
    load_from_cache = '--load' in sys.argv or input('Do you want to load DQN from cache? (y or n): ') == 'y'

    if load_from_cache:
        try:
            Q.load_state_dict(torch.load(CACHE_FILE))
            Q_target.load_state_dict(Q.state_dict())
            print(f"Loaded DQN from '{CACHE_FILE}'")
        except FileNotFoundError:
            print(f"No cache file found at '{CACHE_FILE}'")

    try:
        main(train = train, play = play)
    except KeyboardInterrupt:
        if train and input('Do you want to cache the DQN? (y or n): ') == 'y':
            torch.save(Q.state_dict(), CACHE_FILE)
            print(f"Successfully cached DQN to '{CACHE_FILE}'")

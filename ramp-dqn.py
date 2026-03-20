import gymnasium as gym
import sys
from collections import deque
import random
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

buffer_size = 10_000
batch_size = 32
gamma = 0.99

epsilon_start = 0.9
epsilon_min = 0.01
epsilon_decay = 0.99995
learning_rate = 0.00025
TARGET_SYNC_EVERY = 500

CACHE_FILE = 'ramp_dqn_cache.pt'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

class SimpleNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

Q = SimpleNet(2, 3).to(device)
Q_target = SimpleNet(2, 3).to(device)
Q_target.load_state_dict(Q.state_dict())
optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

def determine_action(state, epsilon):
    with torch.no_grad():
        q_values = Q(state.unsqueeze(0))
    
    if random.random() < epsilon:
        action = random.choice([0, 1])
        return action, q_values.gather(1, torch.tensor([[action]])).item()
    
    q_value, action = q_values.max(dim=1)
    return action.item(), q_value.item()

def train_step(transitions):
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.stack(states)
    actions = torch.tensor(actions,dtype=torch.long)
    rewards = torch.tensor(rewards,dtype=torch.float32)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones,dtype=torch.float32)

    # Q-value for the action actually taken
    current_q = Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target — no gradient through this
    with torch.no_grad():
        max_next_q = Q_target(next_states).max(dim=1).values
        target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(Q.parameters(), 100)
    optimizer.step()

    return loss.detach().item()

def main(train = True):
    env = gym.make("MountainCar-v0", render_mode="human" if not train else None)
    if not train:
        env = env.unwrapped
    
    epsilon = epsilon_start if train else 0
    D = deque([], maxlen=buffer_size)
    episode = 0

    # Plotting
    episode_reward_history = []
    average_q_value_history = []
    loss_history = []
    epsilon_history = []

    average_episode_reward = 0
    average_q_value = 0
    steps_since_inform_user = 0

    train_loss_sum = 0.0
    train_loss_steps = 0
    train_steps = 0

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DQN Training")
    
    while True:
        episode += 1
        obs, _info = env.reset()
        state = torch.tensor(obs,dtype=torch.float32)

        total_reward = 0

        while True:
            steps_since_inform_user += 1

            # determine action
            action, q_value = determine_action(state, epsilon)
            average_q_value += (q_value - average_q_value) / steps_since_inform_user

            # take action
            obs, reward, terminated, truncated, info = env.step(action)
            new_state = torch.tensor(obs,dtype=torch.float32)
            total_reward += reward
            done = terminated or truncated

            # store transition
            transition = (state, action, reward, new_state, done)
            D.append(transition)
            state = new_state

            # gradient descent
            if train and len(D) == buffer_size:
                transitions = random.sample(D, batch_size)
                step_loss = train_step(transitions)

                train_loss_sum += step_loss
                train_loss_steps += 1
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                train_steps += 1

                if train_steps % TARGET_SYNC_EVERY == 0:
                    Q_target.load_state_dict(Q.state_dict())
            
            if done:
                break
        
        average_episode_reward += total_reward / 20
        if episode % 20 == 0:
            # plotting
            episode_reward_history.append(average_episode_reward)
            average_q_value_history.append(average_q_value)
            if train_loss_steps > 0:
                loss_history.append(train_loss_sum / train_loss_steps)
            
            epsilon_history.append(epsilon)

            axes[0,0].cla()
            axes[0,0].plot(episode_reward_history)
            axes[0,0].set_title("Episode Reward")
            axes[0,0].set_xlabel("20 Episodes")

            axes[0,1].cla()
            axes[0,1].plot(average_q_value_history)
            axes[0,1].set_title("Avg Q Value")
            axes[0,1].set_xlabel("20 Episodes")

            axes[1,0].cla()
            axes[1,0].plot(loss_history)
            axes[1,0].set_title("Loss")
            axes[1,0].set_xlabel("20 Episodes")

            axes[1,1].cla()
            axes[1,1].plot(epsilon_history)
            axes[1,1].set_title("Epsilon")
            axes[1,1].set_xlabel("20 Episodes")

            plt.tight_layout()
            plt.pause(0.001)

            avg_loss_str = f"{train_loss_sum / train_loss_steps:.6f}" if train_loss_steps > 0 else "n/a"
            print(
                f"Episode {episode} | Avg reward {average_episode_reward:.2f} | "
                f"Loss {avg_loss_str} | Buffer {len(D)} | ε {epsilon:.3f}"
            )
            
            average_episode_reward = 0
            average_q_value = 0
            steps_since_inform_user = 0

            train_loss_sum = 0.0
            train_loss_steps = 0

if __name__ == "__main__":
    if input('Do you want to load Q from cache (y or n): ') == 'y':
        try:
            Q.load_state_dict(torch.load(CACHE_FILE))
            Q_target.load_state_dict(Q.state_dict())
            print(f"Loaded Q from {CACHE_FILE}")
        except FileNotFoundError:
            print(f"No cache file found at {CACHE_FILE}")

    train = '--train' in sys.argv
    try:
        main(train = train)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if train and input('Do you want to cache the Q algorithm (y or n):') == 'y':
            torch.save(Q.state_dict(), CACHE_FILE)
            print(f"Cached Q to {CACHE_FILE}")

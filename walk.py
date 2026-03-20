import gymnasium as gym
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

REPLAY_BUFFER_SIZE = 50_000
BATCH_SIZE = 32

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_TEST = 0.05
EPSILON_DECAY = 0.99995

LEARNING_RATE = 0.00025
GAMMA = 0.99
TARGET_SYNC_EVERY = 500

k = 4

PLOT_EVERY_FRAMES = 10_000

CACHE_FILE = 'walk-dqn.pt'

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

class DQN(nn.Module):
    def __init__(self, n_input, n_out):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_input, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_out)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

Q = DQN(24 * k, 81).to(device)
Q_target = DQN(24 * k, 81).to(device)
Q_target.load_state_dict(Q.state_dict())
optimizer = optim.Adam(Q.parameters(), lr=LEARNING_RATE)

def num_to_action(num):
    res = []
    temp = num
    for _ in range(4):
        res.append((temp % 3) - 1)
        temp //= 3
    return tuple(res)

def determine_action(state, epsilon):
    with torch.no_grad():
        q_values = Q(state.unsqueeze(0)).squeeze(0)
    
    # epsilon greedy
    if random.random() < epsilon:
        action = random.randint(0, 80)
        return action, q_values[action]
    else:
        action = q_values.argmax().item()
        return action, q_values[action]

def train_step(batch):
    states, actions, rewards, new_states, is_terminals = zip(*batch)

    # convert batch to tensors
    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    new_states = torch.stack(new_states).to(device)
    is_terminals = torch.tensor(is_terminals, dtype=torch.bool).to(device)

    # gradient descent
    with torch.no_grad():
        # double dqn
        best_actions = Q(new_states).argmax(dim=1, keepdim=True)
        max_next_q = Q_target(new_states).gather(1, best_actions).squeeze(1)
        targets = torch.where(is_terminals, rewards, rewards + GAMMA * max_next_q)
    
    actual = Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    optimizer.zero_grad()
    loss = nn.SmoothL1Loss()(actual, targets)
    loss.backward()
    torch.nn.utils.clip_grad_value_(Q.parameters(), 100)
    optimizer.step()

    return loss.detach().item()

def main(train):
    env = gym.make('BipedalWalker-v3', render_mode='human' if not train else None)
    print('Initialized environment')

    epsilon = EPSILON_START if train else EPSILON_TEST
    frame = 0
    train_steps = 0

    D = deque([], REPLAY_BUFFER_SIZE)

    # plotting
    writer = SummaryWriter(log_dir='runs/dqn_walk') if train else None
    average_q_value = 0
    average_loss = 0
    average_episode_reward = 0
    frames_since_update_user = 0
    train_steps_since_update_user = 0
    episodes_since_update_user = 0

    while True:
        episodes_since_update_user += 1
        total_episode_reward = 0

        # initialize state/environment
        obs, info = env.reset()
        compounded_state = deque([], maxlen=k)
        compounded_state.append(obs)
        action = 40 # (0, 0, 0, 0)
        actions_since_decision = 0

        state = None
        prev_state = None
        total_reward_from_action = 0

        while True:
            frame += 1
            frames_since_update_user += 1

            # take action
            obs, reward, terminated, truncated, _info = env.step(num_to_action(action))
            prev_action = action
            total_episode_reward += reward
            total_reward_from_action += reward
            actions_since_decision += 1
            is_terminal = terminated or truncated
            compounded_state.append(obs)

            if actions_since_decision == k or is_terminal:
                # add transition
                prev_state = state
                state = torch.tensor([observed for frame_state in compounded_state for observed in frame_state])
                if prev_state is not None:
                    D.append((prev_state, prev_action, total_reward_from_action, state, is_terminal))
                
                total_reward_from_action = 0

                # determine next action
                action, q_value = determine_action(state, epsilon)
                actions_since_decision = 0
                average_q_value += (q_value - average_q_value) / frames_since_update_user
            
            # train DQN
            # if len(D) >= BATCH_SIZE
            if len(D) == REPLAY_BUFFER_SIZE:
                train_steps += 1
                train_steps_since_update_user += 1

                batch = random.sample(D, BATCH_SIZE)
                loss = train_step(batch)
                average_loss += (loss - average_loss) / train_steps_since_update_user

                # update constants / target network
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
                if train_steps % TARGET_SYNC_EVERY == 0:
                    Q_target.load_state_dict(Q.state_dict())

            # plotting
            if frames_since_update_user == PLOT_EVERY_FRAMES:
                # log results
                loss_str = 'n/a' if train_steps_since_update_user == 0 else f'{average_loss:.5f}'
                print(f'Frame {frame} | Avg Q Value: {average_q_value:.2f} | Avg Episode Reward: {average_episode_reward:.2f} | Loss: {loss_str} | Epsilon: {epsilon:.3f}')

                # plot results
                if train_steps_since_update_user > 0:
                    writer.add_scalar('Average Q Value', average_q_value, frame)
                    writer.add_scalar('Average Episode Reward', average_episode_reward, frame)
                    writer.add_scalar('Loss', average_loss, frame)
                    writer.add_scalar('Epsilon', epsilon, frame)

                average_q_value = 0
                average_loss = 0
                frames_since_update_user = 0
                train_steps_since_update_user = 0
            
            if is_terminal:
                break
        
        average_episode_reward += (total_episode_reward - average_episode_reward) / episodes_since_update_user

if __name__ == '__main__':
    train = '--train' in sys.argv
    load = '--load' in sys.argv
    no_load = '--no-load' in sys.argv
    if load and no_load:
        raise Exception('Cannot have tags --load and --no-load')

    if load or (not no_load and input('Do you want to load DQN from cache? (y or n): ') == 'y'):
        try:
            Q.load_state_dict(torch.load(CACHE_FILE))
            Q_target.load_state_dict(Q.state_dict())
            print(f"Loaded DQN from '{CACHE_FILE}'")
        except FileNotFoundError:
            print(f"No cache file found at '{CACHE_FILE}'")
    
    try:
        main(train = train)
    except KeyboardInterrupt:
        if input('Do you want to save DQN to cache? (y or n): ') == 'y':
            torch.save(Q.state_dict(), CACHE_FILE)
            print(f"Successfully cached DQN to '{CACHE_FILE}'")

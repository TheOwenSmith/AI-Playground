import random
import numpy as np
import matplotlib.pyplot as plt

n = 10
action_tasks = 2_000
q = [random.gauss(0, 1) for _ in range(n)]

def epsilon_greedy(epsilon):
  N = np.array([0] * n)
  Q = np.array([0] * n,dtype=np.float64)

  average_reward = 0
  average_rewards_over_time = []
  for t in range(action_tasks):
    action = random.randrange(n) if random.random() < epsilon else np.argmax(Q)

    reward = q[action] + random.gauss()
    N[action] += 1
    Q[action] = Q[action] + 1 / N[action] * (reward - Q[action])
    average_reward += 1 / (t + 1) * (reward - average_reward)
    average_rewards_over_time.append(average_reward)
  
  return average_rewards_over_time

# def run_sim(epsilon, ucb=False):
#   N = np.array([0] * n)
#   Q = np.array([0] * n,dtype=np.float64)

#   sum_rewards = 0
#   average_rewards = []
#   for t in range(1, action_tasks + 1):
#     action = 0
#     if ucb:
#       # epsilon is really c in this case
#       action = np.argmax(Q + epsilon * np.sqrt(np.log(t) / N))
#     elif (random.random() < epsilon):
#       # exploration
#       action = random.randrange(len(Q))
#     else:
#       # exploitation
#       action = np.argmax(Q)
    
#     reward = q[action] + random.gauss(0, 1)
#     sum_rewards += reward
#     N[action] += 1
#     Q[action] = Q[action] + 1 / N[action] * (reward - Q[action])
#     average_rewards.append(sum_rewards / t)
#   print(N,Q)
#   return average_rewards

def softmax(logits):
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities

def gradient_ascent(step_size):
  H = np.zeros(n, dtype=float)
  sum_rewards = 0
  average_rewards = []

  for t in range(1, action_tasks + 1):
    probabilities = softmax(H)
    A_t = np.random.choice(n, p=probabilities)

    reward = q[A_t] + np.random.normal(0, 1)
    sum_rewards += reward
    average_rewards.append(sum_rewards / t)

    error = reward - average_rewards[-1]
    H -= step_size * error * probabilities
    H[A_t] += step_size * error
  return average_rewards

xpoints = np.array(range(1, action_tasks + 1))
for epsilon in [0, 0.01, 0.1, 0.5]:
    ypoints = np.array(epsilon_greedy(epsilon))
    plt.plot(xpoints, ypoints, label=f'ε = {epsilon}')

# for c in [1, 2]:
#     ypoints = np.array(run_sim(c, ucb=True))
#     plt.plot(xpoints, ypoints, label=f'c = {c}')

for alpha in [0.1, 0.2, 0.4]:
  ypoints = np.array(gradient_ascent(alpha))
  plt.plot(xpoints, ypoints, label=f'alpha = {alpha}')

plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.axhline(y=max(q), color='black', linestyle='--', label=f'Max(q) = {max(q)}')
plt.title('Average Rewards Over Time for Different Epsilon Values')
plt.legend()
plt.show()

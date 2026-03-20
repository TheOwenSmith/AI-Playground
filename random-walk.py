import random
import numpy as np
import matplotlib.pyplot as plt

def generate_episode(start_state = 3):
    episode = []
    state = start_state
    while state not in (0, 6):
        action = random.choice([-1, 1])
        new_state = state + action
        reward = 1 if new_state == 6 else 0
        episode.append((state, reward))
        state = new_state
    return episode

optimal = np.array([i / 6 for i in range(6)] + [0])
def monte_carlo_prediction(n, alpha, gamma=1):
    V = np.ones(7,dtype=np.float64)
    V[0], V[6] = 0, 0

    errors = []
    for _ in range(n):
        episode = generate_episode()

        G = 0
        visited = {}
        for state, reward in reversed(episode):
            G = reward + gamma * G  # γ=1, so just accumulate
            visited[state] = G  # first-visit: overwrite keeps first occurrence

        # Update value estimates
        for state, G in visited.items():
            V[state] += alpha * (G - V[state])
        
        errors.append(root_mean_squared_error(V, optimal))

    return errors

def td_prediction(n, alpha, gamma=1):
    V = np.full(7,0.5,dtype=np.float64)
    V[0], V[6] = 0, 0
    start_state = 3

    errors = []
    for _ in range(n):
        state = start_state
        while state not in (0, 6):
            action = random.choice([-1, 1])
            new_state = state + action
            reward = 1 if new_state == 6 else 0

            V[state] = V[state] + alpha * (reward + gamma * V[new_state] - V[state])
            state = new_state
        
        errors.append(root_mean_squared_error(V, optimal))
    
    return errors

def root_mean_squared_error(have, expected):
    return np.sqrt(np.mean((have - expected) ** 2))

for alpha in [0.01, 0.1, 0.5]:
    # errors_monte_carlo = monte_carlo_prediction(100, alpha)
    errors_td = td_prediction(100, alpha)
    # plt.plot(range(100), errors_monte_carlo, label=f'Monte Carlo (α={alpha})')
    plt.plot(range(100), errors_td, label=f'Temporal Difference (α={alpha})')

plt.title('Monte Carlo vs Temporal Difference')
plt.xlabel('RMSE')
plt.ylabel('Value')
plt.legend()
plt.show()

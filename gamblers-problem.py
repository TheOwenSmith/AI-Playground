import numpy as np
import matplotlib.pyplot as plt

gamma = 1
policy = np.ones(99,dtype=np.int32)
p_h = 0.4

def eval_state_action(capital, action, v):
    # betting outcomes
    successful_capital = capital + action
    unsuccesful_capital = capital - action
    
    reward_successful = 1 if successful_capital == 100 else gamma * v[successful_capital - 1]
    reward_unsuccesful = 0 if unsuccesful_capital == 0 else gamma * v[unsuccesful_capital - 1]
    state_value = p_h * reward_successful + (1 - p_h) * reward_unsuccesful
    return state_value

def eval_policy(theta = 1e-10):
    v = np.zeros(99,dtype=np.float64)

    delta = theta
    while delta >= theta:
        delta = 0
        # state
        for capital in range(1, 100):
            action = policy[capital - 1]
            state_value = eval_state_action(capital, action, v)
            delta = max(delta, abs(state_value - v[capital - 1]))
            v[capital - 1] = state_value

    return v

def print_policy():
    for capital in range(1, 100):
        print(f'{capital} -> {policy[capital - 1]}')

def print_value_function(v):
    for capital in range(1, 100):
        print(f'{capital} -> {v[capital - 1]}')

def find_optimal_policy():
    k = 0
    policy_is_optimal = False
    while not policy_is_optimal:
        v = eval_policy()
        print(f'Iteration {k}')

        policy_is_optimal = True # give benefit of doubt
        for capital in range(1, 100):
            optimal_action = -1
            optimal_state_value = v[capital - 1]
            for action in range(1, min(capital, 100 - capital) + 1):
                # betting outcomes
                state_value_given_action = eval_state_action(capital, action, v)
                if state_value_given_action > optimal_state_value:
                    optimal_state_value = state_value_given_action
                    optimal_action = action
            
            if optimal_state_value > v[capital - 1] + 1e-9:
                policy[capital - 1] = optimal_action
                policy_is_optimal = False
        k += 1
    
    print(f'Optimal policy found. Took {k} iterations')
    print('--Policy--')
    print_policy()
    print('--Value Function--')
    v = eval_policy()
    print_value_function(v)

    # plot

    plt.plot(np.arange(1, 100), policy)
    plt.show()
    plt.plot(np.arange(1, 100), v)
    plt.show()


if __name__ == '__main__':
    find_optimal_policy()

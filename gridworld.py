import numpy as np

down = [1, 0]
up = [-1, 0]
right = [0, 1]
left = [0, -1]

down_arrow = '↓'
up_arrow = '↑'
right_arrow = '→'
left_arrow = '←'

actions = [right, left, down, up]
action_char = [right_arrow, left_arrow, down_arrow, up_arrow]
gamma = 1

def step(i, j, di, dj):
  ni, nj = i + di, j + dj
  if ni < 0 or ni > 3 or nj < 0 or nj > 3:
    return i, j
  return ni, nj

# Evaluate random policy
def eval_policy(policy, theta = 0.001):
  vs = np.zeros((4, 4))
  delta = theta

  while delta >= theta:
    delta = 0
    vs_new = np.zeros((4, 4))
    for i in range(4):
      for j in range(4):
        # terminal states
        if (i == 0 and j == 0) or (i == 3 and j == 3):
          continue
        
        for actionInd in range(len(actions)):
          di, dj = actions[actionInd]
          ni, nj = step(i, j, di, dj)
          # \pi(a|s) * (r+\gamma v(s'))
          vs_new[i][j] += policy(actionInd, (i, j)) * (-1 + gamma * vs[ni][nj])
      
        delta = max(delta, abs(vs_new[i][j] - vs[i][j]))

    vs = vs_new
  
  return vs.round(1)

def print_policy(policy):
  for i in range(4):
    print('\n')
    for j in range(4):
      action = np.argmax(policy[i][j])
      print(action_char[action], end=' ')

def find_optimal_policy():
  # initial random policy
  policy = np.full((4, 4, len(actions)), 1 / len(actions))
  policy_fn = lambda actionInd, state: policy[state[0]][state[1]][actionInd]

  # improve policy iteratively
  policy_stable = False
  while not policy_stable:
    policy_stable = True
    # evaluate policy
    state_value = eval_policy(policy_fn)

    # improve policy
    for i in range(4):
      for j in range(4):
        def q(actionInd):
          di, dj = actions[actionInd]
          ni, nj = step(i, j, di, dj)
          return -1 + gamma * state_value[ni][nj]

        optimal_state_value = max([q(actionInd) for actionInd in range(len(actions))])
        optimal_action = np.argmax([q(actionInd) for actionInd in range(len(actions))])
        
        if optimal_state_value > state_value[i][j]:
          new_action_given_state = np.zeros(4)
          new_action_given_state[optimal_action] = 1

          policy[i][j] = new_action_given_state
          policy_stable = False
  
  return policy

policy = find_optimal_policy()
print_policy(policy)

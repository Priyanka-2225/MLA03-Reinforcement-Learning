import numpy as np

states = [0, 1, 2]  # Low, Medium, High
actions = [0, 1]    # Short green, Long green

gamma = 0.9

# Transition probabilities P[s][a][s']
P = {
    0: {0: [0.7, 0.3, 0.0], 1: [0.8, 0.2, 0.0]},
    1: {0: [0.2, 0.6, 0.2], 1: [0.3, 0.6, 0.1]},
    2: {0: [0.0, 0.4, 0.6], 1: [0.1, 0.6, 0.3]}
}

# Reward = -waiting_time
R = {0: -1, 1: -5, 2: -10}

policy = np.zeros(3, dtype=int)
V = np.zeros(3)

def policy_evaluation():
    global V
    for _ in range(100):
        new_V = np.zeros(3)
        for s in states:
            a = policy[s]
            new_V[s] = R[s] + gamma * sum(
                P[s][a][s2] * V[s2] for s2 in states
            )
        V = new_V

def policy_improvement():
    global policy
    stable = True
    for s in states:
        action_values = []
        for a in actions:
            value = R[s] + gamma * sum(
                P[s][a][s2] * V[s2] for s2 in states
            )
            action_values.append(value)
        best_action = np.argmax(action_values)
        if best_action != policy[s]:
            stable = False
        policy[s] = best_action
    return stable

while True:
    policy_evaluation()
    if policy_improvement():
        break

print("Optimal Policy:", policy)

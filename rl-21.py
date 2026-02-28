import numpy as np

# States: 0=Low, 1=Medium, 2=High
states = 3
actions = 3

gamma = 0.9

# Reward matrix R[s][a]
R = np.array([
    [-5, -10, -20],     # Low traffic
    [-15, -8, -10],     # Medium traffic
    [-30, -15, -5]      # High traffic
])

# Transition probabilities P[s][a][s']
P = {
    0: {0:[0.7,0.3,0.0], 1:[0.6,0.4,0.0], 2:[0.5,0.5,0.0]},
    1: {0:[0.2,0.6,0.2], 1:[0.3,0.5,0.2], 2:[0.4,0.4,0.2]},
    2: {0:[0.0,0.3,0.7], 1:[0.0,0.4,0.6], 2:[0.0,0.5,0.5]}
}

# Initialize policy randomly
policy = np.zeros(states, dtype=int)
V = np.zeros(states)

def policy_evaluation():
    global V
    for _ in range(50):
        newV = np.copy(V)
        for s in range(states):
            a = policy[s]
            newV[s] = R[s][a] + gamma * sum(
                P[s][a][s1] * V[s1] for s1 in range(states)
            )
        V = newV

def policy_improvement():
    global policy
    stable = True
    for s in range(states):
        old_action = policy[s]
        action_values = []
        for a in range(actions):
            value = R[s][a] + gamma * sum(
                P[s][a][s1] * V[s1] for s1 in range(states)
            )
            action_values.append(value)
        policy[s] = np.argmax(action_values)
        if old_action != policy[s]:
            stable = False
    return stable

# Policy Iteration
while True:
    policy_evaluation()
    if policy_improvement():
        break

print("Optimal Policy (0=Short,1=Medium,2=Long):", policy)
print("Optimal State Values:", V)

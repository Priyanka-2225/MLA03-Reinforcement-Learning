import numpy as np

states = 16
actions = 4
gamma = 0.9

P = np.zeros((states, actions, states))
R = np.ones((states, actions, states)) * -1

for s in range(states):
    for a in range(actions):
        next_s = s
        P[s,a,next_s] = 1

policy = np.zeros(states, dtype=int)
V = np.zeros(states)

for _ in range(50):
    # Policy Evaluation
    for _ in range(10):
        for s in range(states):
            V[s] = sum([P[s,policy[s],s2]*(R[s,policy[s],s2]+gamma*V[s2]) for s2 in range(states)])
    # Policy Improvement
    for s in range(states):
        policy[s] = np.argmax([sum([P[s,a,s2]*(R[s,a,s2]+gamma*V[s2]) for s2 in range(states)]) for a in range(actions)])

print("Optimal Policy:", policy)

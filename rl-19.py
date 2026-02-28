import numpy as np

grid = 3
V = np.zeros((grid,grid))
gamma=0.9

policy = {(0,0):(0,1),(0,1):(0,2),(0,2):(1,2),(1,2):(2,2)}

for _ in range(50):
    for state, next_state in policy.items():
        V[state]= -1 + gamma*V[next_state]

print(V)

import numpy as np

grid = 4
gamma = 0.9
V = np.zeros((grid,grid))
goal = (3,3)

for iteration in range(100):
    newV = np.copy(V)
    for i in range(grid):
        for j in range(grid):
            if (i,j)==goal:
                continue
            actions = []
            for move in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+move[0], j+move[1]
                if 0<=ni<grid and 0<=nj<grid:
                    actions.append(-1 + gamma*V[ni,nj])
            newV[i,j] = max(actions)
    V = newV

print(V)

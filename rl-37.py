import numpy as np
import random

grid = 4
Q = np.zeros((grid,grid,4))
alpha=0.1
gamma=0.9
epsilon=0.1

actions=[(0,1),(0,-1),(1,0),(-1,0)]

def choose_action(x,y):
    if random.random()<epsilon:
        return random.randint(0,3)
    return np.argmax(Q[x,y])

for episode in range(2000):
    x,y=0,0
    a=choose_action(x,y)
    
    while (x,y)!=(3,3):
        dx,dy=actions[a]
        nx,ny=x+dx,y+dy
        if not (0<=nx<grid and 0<=ny<grid):
            nx,ny=x,y
        
        reward=1
        if (nx,ny)==(3,3):
            reward=10
        
        a2=choose_action(nx,ny)
        Q[x,y,a]+=alpha*(reward+gamma*Q[nx,ny,a2]-Q[x,y,a])
        
        x,y,a=nx,ny,a2

print("Training done")

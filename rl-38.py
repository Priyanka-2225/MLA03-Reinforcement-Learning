import numpy as np
import random

grid=5
Q=np.zeros((grid,grid,4))
alpha=0.1
gamma=0.9
epsilon=0.1

food=(4,4)
ghost=(2,2)

actions=[(0,1),(0,-1),(1,0),(-1,0)]

for episode in range(3000):
    x,y=0,0
    while (x,y)!=food:
        if random.random()<epsilon:
            a=random.randint(0,3)
        else:
            a=np.argmax(Q[x,y])
        
        dx,dy=actions[a]
        nx,ny=x+dx,y+dy
        if not (0<=nx<grid and 0<=ny<grid):
            nx,ny=x,y
        
        reward=-1
        if (nx,ny)==food:
            reward=20
        if (nx,ny)==ghost:
            reward=-20
        
        Q[x,y,a]+=alpha*(reward+gamma*np.max(Q[nx,ny])-Q[x,y,a])
        x,y=nx,ny

print("Game training complete")

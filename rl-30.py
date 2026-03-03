import numpy as np

grid_size = 5
q_table = np.zeros((grid_size, grid_size, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.1

goal = (4,4)

def step(state, action):
    x,y = state
    if action==0 and x>0: x-=1
    if action==1 and x<4: x+=1
    if action==2 and y>0: y-=1
    if action==3 and y<4: y+=1
    reward = -1
    if (x,y)==goal:
        reward = 10
    return (x,y), reward

for episode in range(500):
    state = (0,0)
    while state!=goal:
        if np.random.rand()<epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state])
        next_state, reward = step(state, action)
        q_table[state][action] += alpha * (reward + gamma*np.max(q_table[next_state]) - q_table[state][action])
        state = next_state

print("Training Complete")

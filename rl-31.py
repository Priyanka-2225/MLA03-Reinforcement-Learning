import numpy as np
import random

q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.1

def get_q(state, action):
    return q.get((state, action), 0.0)

def choose_action(state, actions):
    if random.random() < epsilon:
        return random.choice(actions)
    qs = [get_q(state,a) for a in actions]
    return actions[np.argmax(qs)]

def update(state, action, reward, next_state, next_action):
    predict = get_q(state, action)
    target = reward + gamma * get_q(next_state, next_action)
    q[(state,action)] = predict + alpha*(target-predict)

print("SARSA TicTacToe Skeleton Ready")

import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

state_size=4
action_size=2

model=Sequential([
    Dense(24,input_dim=state_size,activation='relu'),
    Dense(24,activation='relu'),
    Dense(action_size,activation='linear')
])

model.compile(loss='mse',optimizer=Adam(0.001))

memory=deque(maxlen=2000)
gamma=0.95
epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.995

print("DQN model created successfully")

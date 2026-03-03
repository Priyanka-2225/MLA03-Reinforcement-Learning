import numpy as np
import random
import tensorflow as tf
from collections import deque

# Generate simple stock prices
prices = np.sin(np.arange(0, 1000) * 0.01) * 10 + 100

state_size = 1
action_size = 3  # Buy, Sell, Hold

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
episodes = 10

memory = deque(maxlen=2000)

# Build Model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(state_size,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

def act(state):
    global epsilon
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    return np.argmax(model.predict(state, verbose=0)[0])

def replay():
    global epsilon
    
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
        target = reward
        
        if not done:
            next_action = np.argmax(model.predict(next_state, verbose=0)[0])
            target = reward + gamma * target_model.predict(next_state, verbose=0)[0][next_action]
        
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    target_model.set_weights(model.get_weights())

# ----------------------
# Training Loop
# ----------------------

for e in range(episodes):
    total_profit = 0
    
    for t in range(len(prices)-1):
        state = np.array([[prices[t]]])
        next_state = np.array([[prices[t+1]]])
        
        action = act(state)
        
        # Reward logic
        if action == 0:  # Buy
            reward = prices[t+1] - prices[t]
        elif action == 1:  # Sell
            reward = prices[t] - prices[t+1]
        else:  # Hold
            reward = 0
        
        done = (t == len(prices)-2)
        
        memory.append((state, action, reward, next_state, done))
        
        total_profit += reward
        
        replay()
    
    print(f"Episode {e+1}/{episodes} - Total Profit: {round(total_profit,2)} - Epsilon: {round(epsilon,2)}")

print("Training Finished!")

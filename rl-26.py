import numpy as np
import random
import gymnasium as gym
import tensorflow as tf
from collections import deque

# Create environment
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory = deque(maxlen=2000)

# Model (Correct Modern Style)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(state_size,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    global epsilon
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

def replay():
    global epsilon
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state, verbose=0)[0])
        
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        
        model.fit(state, target_f, epochs=1, verbose=0)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

episodes = 200

for e in range(episodes):
    state, _ = env.reset()   # ✅ Gymnasium reset format
    state = np.reshape(state, [1, state_size])
    
    for time in range(500):
        action = act(state)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated   # ✅ Important Fix
        
        next_state = np.reshape(next_state, [1, state_size])
        
        remember(state, action, reward, next_state, done)
        state = next_state
        
        replay()
        
        if done:
            print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {epsilon:.2f}")
            break

print("Training Finished!")
env.close()

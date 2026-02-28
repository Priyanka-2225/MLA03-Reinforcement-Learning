import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Create MountainCar Environment
env = gym.make("MountainCar-v0")

# Build Neural Network (No warning version)
model = tf.keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(3, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 300

for episode in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, 2])
    total_reward = 0
    done = False

    while not done:
        # Epsilon-Greedy Strategy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, [1, 2])

        target = reward
        if not done:
            target += gamma * np.max(model.predict(next_state, verbose=0)[0])

        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target

        model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state
        total_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

print("Training Completed Successfully!")
env.close()

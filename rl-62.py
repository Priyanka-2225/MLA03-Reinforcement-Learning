import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class StoryEnv:
    def reset(self):
        self.creativity = np.random.uniform(0, 1)
        return np.array([self.creativity], dtype=np.float32)

    def step(self, action):
        if action == 0:  # low effort
            engagement = np.random.uniform(0.1, 0.5)
        else:  # high effort
            engagement = np.random.uniform(0.5, 1.0)

        reward = engagement
        return np.array([engagement], dtype=np.float32), reward


model = tf.keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(1,)),
    layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.01)

env = StoryEnv()
gamma = 0.95

for episode in range(200):
    state = env.reset()
    state_input = state.reshape(1, -1)

    with tf.GradientTape() as tape:
        probs = model(state_input)
        action = np.random.choice(2, p=probs.numpy()[0])
        next_state, reward = env.step(action)

        loss = -tf.math.log(probs[0][action]) * reward

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if episode % 20 == 0:
        print("Episode:", episode, "Engagement Reward:", reward)

print("Storytelling Policy Gradient Training Complete")

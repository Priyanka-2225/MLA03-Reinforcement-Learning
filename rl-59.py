import numpy as np
import tensorflow as tf

prices = np.sin(np.arange(0,200)*0.1)*10+100

class TradingEnv:
    def reset(self):
        self.step_count = 0
        return np.array([prices[self.step_count]])

    def step(self, action):
        self.step_count += 1
        reward = action * (prices[self.step_count] - prices[self.step_count-1])
        return np.array([prices[self.step_count]]), reward

actor = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1, activation='tanh')
])

optimizer = tf.keras.optimizers.Adam(0.01)
env = TradingEnv()

for episode in range(50):
    state = env.reset().reshape(1,-1)
    with tf.GradientTape() as tape:
        action = actor(state)
        _, reward = env.step(action[0][0])
        loss = -reward
    grads = tape.gradient(loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables))

print("DDPG Trading Complete")

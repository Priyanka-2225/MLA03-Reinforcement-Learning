import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Generate synthetic stock data
prices = np.sin(np.arange(0, 500) * 0.05) * 10 + 100

class TradingEnv:
    def reset(self):
        self.t = 0
        return np.array([prices[self.t]], dtype=np.float32)

    def step(self, action):
        self.t += 1
        done = self.t >= len(prices) - 1

        price_diff = prices[self.t] - prices[self.t - 1]

        if action == 0:  # buy
            reward = price_diff
        else:  # sell
            reward = -price_diff

        return np.array([prices[self.t]], dtype=np.float32), reward, done


model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(1,)),
    layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.001)
env = TradingEnv()
gamma = 0.99

for episode in range(200):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_input = state.reshape(1, -1)

        with tf.GradientTape() as tape:
            probs = model(state_input)
            action = np.random.choice(2, p=probs.numpy()[0])
            next_state, reward, done = env.step(action)

            loss = -tf.math.log(probs[0][action]) * reward

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        state = next_state
        total_reward += reward

    if episode % 20 == 0:
        print("Episode:", episode, "Profit:", total_reward)

print("Automated Trading REINFORCE Training Complete")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Simple finance environment
class FinanceEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.savings = 1000
        self.expenses = 500
        self.investment = 0.2
        return np.array([self.savings, self.expenses, self.investment])

    def step(self, action):
        if action == 0:  # Save more
            self.savings += 50
        elif action == 1:  # Invest more
            self.investment += 0.05
        elif action == 2:  # Reduce expenses
            self.expenses -= 20

        profit = self.savings * self.investment * 0.01
        reward = profit - self.expenses * 0.001
        return np.array([self.savings, self.expenses, self.investment]), reward

# Simple PPO-style network
model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(3,)),
    layers.Dense(3, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.001)

env = FinanceEnv()

for episode in range(50):
    state = env.reset()
    state = state.reshape(1, -1)
    with tf.GradientTape() as tape:
        probs = model(state)
        action = np.random.choice(3, p=probs.numpy()[0])
        _, reward = env.step(action)
        loss = -tf.math.log(probs[0][action]) * reward
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Finance PPO Training Complete")

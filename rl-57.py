import numpy as np
import tensorflow as tf

ideal_temp = 24

class HomeEnv:
    def reset(self):
        self.temp = np.random.randint(18, 30)
        return np.array([self.temp])

    def step(self, action):
        if action == 0:
            self.temp -= 1
        else:
            self.temp += 1
        reward = -abs(self.temp - ideal_temp)
        return np.array([self.temp]), reward

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.01)
env = HomeEnv()

for episode in range(100):
    state = env.reset().reshape(1, -1)
    with tf.GradientTape() as tape:
        probs = model(state)
        action = np.random.choice(2, p=probs.numpy()[0])
        _, reward = env.step(action)
        loss = -tf.math.log(probs[0][action]) * reward
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Smart Home REINFORCE Complete")

import numpy as np
import tensorflow as tf

class SoccerEnv:
    def reset(self):
        self.distance = 10
        return np.array([self.distance])

    def step(self, action):
        if action == 0:  # move forward
            self.distance -= 1
        reward = 1 if self.distance <= 0 else -0.1
        return np.array([self.distance]), reward

actor = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

critic = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(0.01)
env = SoccerEnv()

for episode in range(100):
    state = env.reset().reshape(1, -1)
    with tf.GradientTape() as tape:
        probs = actor(state)
        value = critic(state)
        action = np.random.choice(2, p=probs.numpy()[0])
        next_state, reward = env.step(action)
        advantage = reward - value
        actor_loss = -tf.math.log(probs[0][action]) * advantage
        critic_loss = advantage**2
        loss = actor_loss + critic_loss
    grads = tape.gradient(loss, actor.trainable_variables + critic.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables + critic.trainable_variables))

print("Soccer A3C Training Done")

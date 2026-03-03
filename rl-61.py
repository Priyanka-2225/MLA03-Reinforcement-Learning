import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Racing Environment
class RacingEnv:
    def __init__(self):
        self.track_length = 50

    def reset(self):
        self.position = 0
        return np.array([self.position], dtype=np.float32)

    def step(self, action):
        if action == 0:  # accelerate
            self.position += 2
        else:  # brake
            self.position += 1

        reward = self.position / self.track_length
        done = self.position >= self.track_length
        return np.array([self.position], dtype=np.float32), reward, done


# Actor-Critic Model
actor = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(1,)),
    layers.Dense(2, activation='softmax')
])

critic = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(1,)),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(0.001)
env = RacingEnv()
gamma = 0.99

for episode in range(200):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_input = state.reshape(1, -1)

        with tf.GradientTape() as tape:
            probs = actor(state_input)
            value = critic(state_input)

            action = np.random.choice(2, p=probs.numpy()[0])
            next_state, reward, done = env.step(action)

            next_value = critic(next_state.reshape(1, -1))
            advantage = reward + gamma * next_value - value

            actor_loss = -tf.math.log(probs[0][action]) * tf.stop_gradient(advantage)
            critic_loss = advantage ** 2

            loss = actor_loss + critic_loss

        grads = tape.gradient(loss, actor.trainable_variables + critic.trainable_variables)
        optimizer.apply_gradients(zip(grads,
                                      actor.trainable_variables + critic.trainable_variables))

        state = next_state
        total_reward += reward

    if episode % 20 == 0:
        print("Episode:", episode, "Reward:", total_reward)

print("Autonomous Racing A2C Training Complete")

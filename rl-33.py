import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

state_size = 4
action_size = 2

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(state_size,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(action_size, activation='softmax')
])

print("Basic PPO Policy Network Built")

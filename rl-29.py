import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random

state_size = 4
action_size = 4

def build_dueling_model():
    inputs = layers.Input(shape=(state_size,))
    x = layers.Dense(24, activation='relu')(inputs)
    x = layers.Dense(24, activation='relu')(x)

    value = layers.Dense(1)(x)
    advantage = layers.Dense(action_size)(x)

    q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_dueling_model()
print("Dueling DQN Model Built Successfully")

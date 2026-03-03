import numpy as np
from tensorflow.keras.models import clone_model

prices=np.sin(np.arange(0,1000)*0.01)*10+100

model_target=clone_model(model)
model_target.set_weights(model.get_weights())

def update_target():
    model_target.set_weights(model.get_weights())

print("Double DQN structure ready")

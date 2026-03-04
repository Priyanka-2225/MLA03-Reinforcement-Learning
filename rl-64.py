import numpy as np
import matplotlib.pyplot as plt

# Target position
target = np.array([3, 3, 3])

# Initialize joint parameters
theta = np.random.randn(3)
lr = 0.01

def forward_model(theta):
    return theta  # simple identity mapping

def loss(theta):
    pos = forward_model(theta)
    return np.sum((pos - target) ** 2)

def gradient(theta):
    return 2 * (forward_model(theta) - target)

losses = []

for i in range(200):
    grad = gradient(theta)
    theta -= lr * grad
    losses.append(loss(theta))

plt.plot(losses)
plt.title("Robotic Arm Optimization")
plt.show()

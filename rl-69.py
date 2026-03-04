import numpy as np
import matplotlib.pyplot as plt

pos = np.array([0.0, 0.0])
goal = np.array([5.0, 5.0])
lr = 0.1

path = [pos.copy()]

for i in range(50):
    grad = 2 * (pos - goal)
    pos -= lr * grad
    path.append(pos.copy())

path = np.array(path)
plt.plot(path[:,0], path[:,1])
plt.scatter(*goal, color='red')
plt.show()

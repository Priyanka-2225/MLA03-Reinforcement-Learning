import numpy as np
import matplotlib.pyplot as plt
import random

start = (0, 0)
goal = (9, 9)
nodes = [start]

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

for i in range(200):
    rand = (random.randint(0, 10), random.randint(0, 10))
    nearest = min(nodes, key=lambda n: distance(n, rand))
    nodes.append(rand)

x, y = zip(*nodes)
plt.scatter(x, y)
plt.scatter(*start, color='green')
plt.scatter(*goal, color='red')
plt.title("Simple RRT Exploration")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# High-level route (waypoints)
route = [(0,0), (5,5), (10,2)]
position = np.array([0.0,0.0])
path = [position.copy()]

# Low-level controller
for waypoint in route:
    waypoint = np.array(waypoint)
    for _ in range(50):
        grad = waypoint - position
        position += 0.05 * grad   # low-level move
        path.append(position.copy())

path = np.array(path)
plt.plot(path[:,0], path[:,1])
plt.scatter(*zip(*route), color='red')
plt.title("Hierarchical Navigation")
plt.show()

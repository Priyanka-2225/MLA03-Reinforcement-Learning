import numpy as np

drones = np.random.rand(3)

for _ in range(20):
    global_reward = np.sum(drones)
    drones += 0.05*(global_reward - drones)

print("Drone states:", drones)

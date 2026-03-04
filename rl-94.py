import numpy as np

swarm = np.random.rand(5)

for _ in range(20):
    avg = np.mean(swarm)
    swarm += 0.1*(avg - swarm)

print("Adapted swarm:", swarm)

import numpy as np

traffic = np.random.randint(10, 100, 24)

for t in traffic:
    green_time = t / 2
    print("Traffic:", t, "Green time:", green_time)

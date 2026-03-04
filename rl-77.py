import numpy as np

demand = np.random.randint(100, 200, 24)
storage = 50

for d in demand:
    production = d - storage
    storage += 5
    print("Production:", production)

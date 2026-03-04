import numpy as np

orders = np.random.poisson(50, 30)
inventory = 100

for o in orders:
    inventory -= o
    if inventory < 20:
        inventory += 80
    print("Inventory:", inventory)

import numpy as np

inventory = 50
holding_cost = 1
stockout_cost = 5

for day in range(30):
    demand = np.random.poisson(20)
    inventory -= demand
    if inventory < 10:
        inventory += 40  # reorder
    
    cost = holding_cost * max(inventory, 0) + stockout_cost * max(-inventory, 0)
    print("Day:", day, "Inventory:", inventory, "Cost:", cost)

import numpy as np

infected = 10
for day in range(20):
    infected += infected * 0.1
    print("Day:", day, "Infected:", int(infected))

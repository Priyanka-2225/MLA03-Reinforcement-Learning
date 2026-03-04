import numpy as np

resources = 50
lr = 0.1

for i in range(50):
    demand = np.random.randint(30, 70)
    grad = demand - resources
    resources += lr * grad

print("Optimized Resources:", resources)

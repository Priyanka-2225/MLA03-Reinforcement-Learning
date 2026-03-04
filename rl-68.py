import numpy as np

price = 10
lr = 0.01

for i in range(100):
    demand = 100 - 5 * price
    revenue = price * demand
    grad = demand - 5 * price
    price += lr * grad

print("Optimized Price:", price)

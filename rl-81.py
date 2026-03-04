import numpy as np

temp = 30
desired = 24
lr = 0.1

for i in range(50):
    grad = temp - desired
    temp -= lr * grad

print("Optimized Temperature:", temp)

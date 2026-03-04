import numpy as np

param = 5.0

for env in [2, 4, 6]:  # different environments
    for _ in range(20):
        grad = param - env
        param -= 0.1 * grad
    print("Adapted Parameter for env", env, ":", param)

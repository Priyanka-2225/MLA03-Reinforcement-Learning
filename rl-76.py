import numpy as np

prices = 100 + np.cumsum(np.random.randn(100))
print("Simulated prices:", prices[:10])

import numpy as np

prices = np.cumsum(np.random.randn(100))
profit = 0

for i in range(1, len(prices)):
    if prices[i] > prices[i-1]:
        profit += prices[i] - prices[i-1]

print("Strategy Profit:", profit)

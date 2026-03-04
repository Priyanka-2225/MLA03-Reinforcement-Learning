import numpy as np

signals = np.random.rand(4)

for _ in range(20):
    congestion = np.sum(signals)
    signals += 0.05*(congestion - signals)

print("Optimized signals:", signals)

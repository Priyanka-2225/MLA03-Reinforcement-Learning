import numpy as np

patients = np.random.poisson(10, 30)
beds = 15

for p in patients:
    treated = min(p, beds)
    reward = treated - 0.5*(beds-treated)
    print("Reward:", reward)

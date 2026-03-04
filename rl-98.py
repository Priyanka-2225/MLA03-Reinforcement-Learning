import numpy as np

preference = np.random.rand()

for performance in [0.8,0.6,0.9]:
    preference += 0.1*(performance - preference)

print("Updated Preference:", preference)

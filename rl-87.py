import numpy as np

belief = 0.5  # probability of being in state A

for obs in [1,0,1,1]:
    belief = 0.7*belief + 0.3*obs
    print("Updated belief:", belief)

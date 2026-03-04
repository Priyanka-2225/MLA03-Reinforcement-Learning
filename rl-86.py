import numpy as np

agents = [0, 0]

for step in range(20):
    reward = sum(agents)
    agents = [a + 0.1*(reward - a) for a in agents]

print("Final Agent States:", agents)

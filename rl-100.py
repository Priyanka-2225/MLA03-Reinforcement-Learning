import numpy as np

belief_position = 0.5

for sensor in [0.6,0.4,0.7]:
    belief_position = 0.7*belief_position + 0.3*sensor
    print("Updated Position Belief:", belief_position)

# ==============================
# IMPORTS (MUST BE FIRST)
# ==============================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# DEFINE ENVIRONMENT CLASS
# ==============================

class DeliveryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.size = 4
        self.goal = (3, 3)
        self.action_space = spaces.Discrete(4)  # 0=Up,1=Down,2=Left,3=Right
        self.observation_space = spaces.Box(low=0, high=3, shape=(2,), dtype=np.int32)

    def step_state(self, state, action):
        x, y = state

        if action == 0 and x > 0: x -= 1
        if action == 1 and x < 3: x += 1
        if action == 2 and y > 0: y -= 1
        if action == 3 and y < 3: y += 1

        return (x, y)

# ==============================
# CREATE ENV AFTER CLASS
# ==============================

env = DeliveryEnv()

# ==============================
# BELLMAN VALUE COMPUTATION
# ==============================

gamma = 0.9
V = np.zeros((4, 4))

for iteration in range(100):
    for i in range(4):
        for j in range(4):

            if (i, j) == env.goal:
                continue

            values = []

            for action in range(4):
                next_state = env.step_state((i, j), action)

                if next_state == env.goal:
                    reward = 10
                else:
                    reward = -1

                values.append(reward + gamma * V[next_state])

            V[i, j] = max(values)

print("Computed Value Function:")
print(V)

# ==============================
# VISUALIZATION
# ==============================

plt.imshow(V)
plt.colorbar()
plt.title("State Value Function")
plt.show()

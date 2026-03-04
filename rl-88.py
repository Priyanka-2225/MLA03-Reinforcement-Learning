import numpy as np

actions = [10, -5]  # reward, harm
ethical_reward = actions[0] - abs(actions[1])*2
print("Ethical adjusted reward:", ethical_reward)

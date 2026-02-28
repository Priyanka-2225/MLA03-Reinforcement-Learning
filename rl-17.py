import numpy as np

arms = 5
true_ctr = np.random.rand(arms)
Q = np.zeros(arms)
N = np.zeros(arms)

for t in range(1,2000):
    ucb = Q + np.sqrt(2*np.log(t+1)/(N+1e-5))
    a = np.argmax(ucb)
    reward = np.random.binomial(1,true_ctr[a])
    N[a]+=1
    Q[a]+= (reward-Q[a])/N[a]

print("Best Arm:", np.argmax(Q))

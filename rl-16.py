import numpy as np

articles = 5
true_ctr = np.random.rand(articles)

Q = np.zeros(articles)
N = np.zeros(articles)
epsilon = 0.1

for i in range(2000):
    if np.random.rand()<epsilon:
        a = np.random.randint(articles)
    else:
        a = np.argmax(Q)
    
    click = np.random.binomial(1,true_ctr[a])
    N[a]+=1
    Q[a]+= (click-Q[a])/N[a]

print("Best Article:", np.argmax(Q))

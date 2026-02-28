import numpy as np

max_stock = 5
gamma=0.9
V = np.zeros(max_stock+1)

for _ in range(100):
    newV = np.copy(V)
    for s in range(max_stock+1):
        costs=[]
        for order in range(max_stock+1-s):
            holding = s*1
            order_cost = order*2
            next_s = s+order-1 if s+order>0 else 0
            costs.append(-(holding+order_cost) + gamma*V[next_s])
        newV[s]=max(costs)
    V=newV

print("Optimal Values:",V)

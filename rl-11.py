import numpy as np

actions = 9
true_rewards = np.random.rand(actions)

def epsilon_greedy(epsilon, steps=1000):
    Q = np.zeros(actions)
    N = np.zeros(actions)
    wins = 0
    
    for t in range(steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(actions)
        else:
            a = np.argmax(Q)
        
        reward = np.random.binomial(1, true_rewards[a])
        N[a] += 1
        Q[a] += (reward - Q[a]) / N[a]
        wins += reward
    return wins

def softmax(temp=0.1, steps=1000):
    Q = np.zeros(actions)
    N = np.zeros(actions)
    wins = 0
    
    for t in range(steps):
        probs = np.exp(Q/temp) / np.sum(np.exp(Q/temp))
        a = np.random.choice(actions, p=probs)
        
        reward = np.random.binomial(1, true_rewards[a])
        N[a] += 1
        Q[a] += (reward - Q[a]) / N[a]
        wins += reward
    return wins

print("Epsilon Greedy Wins:", epsilon_greedy(0.1))
print("Softmax Wins:", softmax())

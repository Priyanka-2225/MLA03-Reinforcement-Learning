import numpy as np

arms = 5
true_prob = np.random.rand(arms)

# Thompson Sampling
success = np.ones(arms)
failure = np.ones(arms)

for i in range(2000):
    samples = np.random.beta(success,failure)
    a = np.argmax(samples)
    reward = np.random.binomial(1,true_prob[a])
    if reward==1:
        success[a]+=1
    else:
        failure[a]+=1

print("Best Arm (Thompson):", np.argmax(success/(success+failure)))

import numpy as np
from sklearn.linear_model import LinearRegression

returns = np.random.randn(100, 1)
future = np.roll(returns, -1)

model = LinearRegression()
model.fit(returns[:-1], future[:-1])

pred = model.predict([[0.02]])
print("Predicted future return:", pred)

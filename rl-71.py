import numpy as np
from sklearn.linear_model import LinearRegression

prices = np.cumsum(np.random.randn(200))
X = prices[:-1].reshape(-1,1)
y = prices[1:]

model = LinearRegression().fit(X, y)
print("Next price prediction:", model.predict([[prices[-1]]]))

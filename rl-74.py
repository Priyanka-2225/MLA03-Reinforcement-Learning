import numpy as np
from sklearn.linear_model import LinearRegression

prices = np.cumsum(np.random.randn(150))
model = LinearRegression().fit(prices[:-1].reshape(-1,1), prices[1:])
print("Next value:", model.predict([[prices[-1]]]))

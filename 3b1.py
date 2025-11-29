import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

np.random.seed(0) 
X = 2 * np.random.rand(100, 1) 
y = 4 + 3 * X + np.random.randn(100, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

lin_reg = LinearRegression() 
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test) 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 

print(f"Intercept: {lin_reg.intercept_}") 
print(f"Coefficient: {lin_reg.coef_}") 
print(f"Mean Squared Error (MSE): {mse}") 
print(f"R-squared: {r2}") 
 
plt.scatter(X_test, y_test, color="blue") 
plt.plot(X_test, y_pred, color="red", linewidth=2) 
plt.title(f"Simple Linear Regression (R-squared = {r2:.2f})") 
plt.xlabel('X Values') 
plt.ylabel('Y Values') 
plt.show()
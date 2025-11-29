import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

data = { 
            'X1': 2 * np.random.rand(100), 
            'X2': 5 * np.random.rand(100), 
            'y': 4 + 3 * np.random.rand(100) + 2 * np.random.randn(100) 
       } 
df = pd.DataFrame(data) 
 
X = df[['X1', 'X2']]  
y = df['y']           

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

lin_reg_mult = LinearRegression() 
lin_reg_mult.fit(X_train, y_train) 
y_pred_mult = lin_reg_mult.predict(X_test) 
mse_mult = mean_squared_error(y_test, y_pred_mult) 
r2_mult = r2_score(y_test, y_pred_mult) 
 
print(f"Intercept: {lin_reg_mult.intercept_}") 
print(f"Coefficients: {lin_reg_mult.coef_}") 
print(f"Mean Squared Error (MSE): {mse_mult}") 
print(f"R-squared: {r2_mult}") 

plt.scatter(y_test, y_pred_mult - y_test) 
plt.axhline(0, color='red', linewidth=2) 
plt.title(f"Residuals of Multiple Linear Regression (R-squared = {r2_mult:.2f})") 
plt.xlabel('Actual Values') 
plt.ylabel('Residuals') 
plt.show() 
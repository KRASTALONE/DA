import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
np.random.seed(0) 
x = np.random.rand(100)  
y = 2 * x + np.random.normal(0, 0.1, 100)  

correlation_coefficient = np.corrcoef(x, y)[0, 1] 
print(f"Correlation Coefficient: {correlation_coefficient}") 
 
plt.figure(figsize=(8, 6)) 
sns.scatterplot(x=x, y=y) 

plt.title(f"Scatter Plot (Correlation Coefficient = {correlation_coefficient:.2f})") 
plt.xlabel('X Values') 
plt.ylabel('Y Values') 
plt.show()
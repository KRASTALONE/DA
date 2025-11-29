
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

np.random.seed(0) 
X = np.random.rand(100, 2)   
y = (X[:, 0] + X[:, 1] > 1).astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
log_reg = LogisticRegression() 
log_reg.fit(X_train, y_train) 
y_pred = log_reg.predict(X_test) 
 
accuracy = accuracy_score(y_test, y_pred) 
conf_matrix = confusion_matrix(y_test, y_pred) 
class_report = classification_report(y_test, y_pred) 

print(f"Accuracy: {accuracy}") 
print("Confusion Matrix:") 
print(conf_matrix) 
print("Classification Report:") 
print(class_report)
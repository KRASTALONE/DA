import pandas as pd 
import statsmodels.api as sm 
from statsmodels.formula.api import ols 
from scipy.stats import chi2_contingency 

data = pd.DataFrame
({ 
    'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'], 
    'values': [12, 13, 14, 15, 16, 15, 10, 11, 12] 
}) 
  
model = ols('values ~ C(group)', data=data).fit() 
anova_table = sm.stats.anova_lm(model, typ=2) 
print(anova_table) 

data = pd.DataFrame
({ 
    'Preference': ['Tea', 'Coffee', 'Tea', 'Coffee', 'Tea', 'Coffee'], 
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Male'] 
}) 
 
contingency_table = pd.crosstab(data['Gender'], data['Preference']) 

chi2, p, dof, expected = chi2_contingency(contingency_table) 
print(f"Chi-Square: {chi2}") 
print(f"P-value: {p}") 
print(f"Degrees of freedom: {dof}") 
print(f"Expected frequencies:\n{expected}")
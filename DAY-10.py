import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame
df_numeric = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = df_numeric.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix - Boston Housing")
plt.show()
target_corr = corr_matrix['MEDV'].drop('MEDV')  # remove target itself
top_features = target_corr.abs().sort_values(ascending=False).head(5)
print("Top 5 features most correlated with MEDV:")
print(top_features)

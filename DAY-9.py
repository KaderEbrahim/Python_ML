import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
boston = fetch_openml(name='boston', version=1, as_frame=True)
df = boston.frame  
print("=== Describe ===")
print(df.describe())
print("\n=== Info ===")
print(df.info())
print("\n=== Missing Values ===")
print(df.isnull().sum())
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['MEDV'])
plt.title("Boxplot of MEDV (Target)")
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['CRIM'])
plt.title("Boxplot of CRIM (Crime Rate)")
plt.show()
features = df.drop(columns=['MEDV']) 
target = df['MEDV']
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
df_standard = pd.DataFrame(scaler_standard.fit_transform(features), columns=[col + "_std" for col in features.columns])
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(features), columns=[col + "_minmax" for col in features.columns])
df_scaled = pd.concat([df[['MEDV']], df_standard, df_minmax], axis=1)
print("\n=== Standard Scaled Features ===")
print(df_standard.describe())
print("\n=== MinMax Scaled Features ===")
print(df_minmax.describe())
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
sns.kdeplot(df['CRIM'], ax=axes[0, 0])
axes[0, 0].set_title("Original CRIM")
sns.kdeplot(df_standard['CRIM_std'], ax=axes[0, 1])
axes[0, 1].set_title("Standard Scaled CRIM")
sns.kdeplot(df['ZN'], ax=axes[1, 0])
axes[1, 0].set_title("Original ZN")
sns.kdeplot(df_minmax['ZN_minmax'], ax=axes[1, 1])
axes[1, 1].set_title("MinMax Scaled ZN")
plt.tight_layout()
plt.show()

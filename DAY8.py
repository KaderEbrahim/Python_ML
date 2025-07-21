import pandas as pd
df = pd.read_csv("iris.csv")
print("Summary Statistics:")
print(df.describe())
print("\nDataFrame Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

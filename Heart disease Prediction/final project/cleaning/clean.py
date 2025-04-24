import pandas as pd


data = pd.read_csv("heart.csv")
print(data)

print(data.isnull().sum())
data.dropna(inplace=True)
print(data.isnull().sum())

data.to_csv("clean.csv",index=False)
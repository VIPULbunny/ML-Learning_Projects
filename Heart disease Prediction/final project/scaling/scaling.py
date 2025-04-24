import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("clean.csv")

features = ["age", "sex", "cp", "trestbps","chol","fbs", "restecg","thalach","exang","oldpeak","slope","ca","thal"]

data2 = StandardScaler()

data[features] = data2.fit_transform(data[features])

data.to_csv("scaled.csv",index=False)

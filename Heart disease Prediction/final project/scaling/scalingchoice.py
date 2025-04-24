import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("clean.csv")

# Select numerical features for scaling
numerical_features = ["age", "sex", "cp", "trestbps","chol","fbs", "restecg","thalach","exang","oldpeak","slope","ca","thal"]

# Plot histograms
df[numerical_features].hist(figsize=(12, 8), bins=20)
plt.show()

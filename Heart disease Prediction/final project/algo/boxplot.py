import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib.ticker import MaxNLocator

df = pd.read_csv("scaled.csv")

X = df.drop(columns=["num"])
y = df["num"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

accuracies = {name: [] for name in models.keys()}

for _ in range(5):  
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100  
        accuracies[name].append(round(accuracy))          

acc_df = pd.DataFrame(accuracies)

plt.figure(figsize=(10, 6))
sns.boxplot(data=acc_df)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.title("Algorithm Comparison")

plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()

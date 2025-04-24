import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

df = pd.read_csv("heart.csv")  

df.fillna(df.mean(), inplace=True)  

feature_1, feature_2 = "age", "chol"  

X = df[[feature_1, feature_2]]  
y = df["target"]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['red', 'blue']
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(cmap_bold), edgecolors='k', label="Train Data")

random_index = np.random.randint(0, len(X_test))
random_point = X_test[random_index]
predicted_label = knn.predict([random_point])[0]

plt.scatter(random_point[0], random_point[1], c='green', marker='*', s=200, edgecolors='k', label="Test Point (Predicted)")

plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title(f"KNN Decision Boundary (k={knn.n_neighbors})")
plt.legend()
plt.show()

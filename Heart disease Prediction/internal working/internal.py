import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

df = pd.read_csv("heart.csv")  

X = df.drop(columns=['target'])  
y = df['target']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = X_train.copy()
train_data['target'] = y_train
train_data.to_csv("train_data.csv", index=False)

test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv("test_data.csv", index=False)

rf_model = RandomForestClassifier(n_estimators=8, random_state=42)  
rf_model.fit(X_train, y_train)

tree_model = rf_model.estimators_[0]  

used_features = set(X.columns[i] for i in tree_model.tree_.feature if i != -2)  # -2 means a leaf node

visualization_data = X_train[list(used_features)].copy()
visualization_data['target'] = y_train
visualization_data.to_csv("visualization.csv", index=False)

plt.figure(figsize=(15, 10))
plot_tree(tree_model, feature_names=X.columns, class_names=['No Heart Disease', 'Heart Disease'], filled=True)
plt.title("Decision Tree from Random Forest")
plt.show()

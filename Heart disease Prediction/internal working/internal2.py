import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("heart.csv")

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save training data
train_data = pd.DataFrame(X_train, columns=df.drop(columns=['target']).columns)
train_data['target'] = y_train.values
train_data.to_csv("train_data.csv", index=False)

# Save testing data
test_data = pd.DataFrame(X_test, columns=df.drop(columns=['target']).columns)
test_data['target'] = y_test.values
test_data.to_csv("test_data.csv", index=False)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=8, random_state=42)
rf_model.fit(X_train, y_train)

# Extract a single decision tree
tree_model = rf_model.estimators_[0]

# Identify used features
used_features = set(df.drop(columns=['target']).columns[i] for i in tree_model.tree_.feature if i != -2)

# Save visualization data
visualization_data = pd.DataFrame(X_train, columns=df.drop(columns=['target']).columns)[list(used_features)]
visualization_data['target'] = y_train.values
visualization_data.to_csv("visualization_data.csv", index=False)

# Plot decision tree
plt.figure(figsize=(15, 10))
plot_tree(tree_model, feature_names=df.drop(columns=['target']).columns, class_names=['No Heart Disease', 'Heart Disease'], filled=True)
plt.title("Decision Tree from Random Forest")
plt.show()

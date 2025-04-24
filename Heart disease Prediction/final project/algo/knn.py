import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score as score, confusion_matrix as matrix, classification_report as report

df = pd.read_csv("scaled.csv")  


X = df.drop(columns=["num"])  
y = df["num"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


accuracy = score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print("Confusion Matrix:\n", matrix(y_test, y_pred))
print("Classification Report:\n", report(y_test, y_pred))




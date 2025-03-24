# %% Import necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting data visualization
import pandas as pd  # For handling and manipulating datasets
import seaborn as sns  # For enhanced visualization

# %% Define dataset
# 31 values representing years of experience (X) and corresponding salaries (Y)
var_x = [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 
         7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5, 11.2, 11.5]
var_y = [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 
         67938, 66029, 83088, 81363, 93940, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872, 127345]

# %% Verify dataset size
print(len(var_x))  # Should output 30
print(len(var_y))  # Should output 30 (both lists should be of equal length)

# %% Scatter plot to visualize relationship between Experience (X) and Salary (Y)
plt.scatter(var_x, var_y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

# %% Convert dataset into a Pandas DataFrame
df = pd.DataFrame({'Experience': var_x, 'Salary': var_y})

# %% Display first five rows of the dataset
df.head()

# %% Display dataset information (column names, non-null count, data types)
df.info()

# %% Reshape X to a 2D array for model training
X = df.Experience.values.reshape(-1,1)  # Converts 1D array to 2D for scikit-learn compatibility
Y = df.Salary  # Target variable (dependent variable)

# %% Import train_test_split function for splitting dataset
from sklearn.model_selection import train_test_split

# %% Split dataset into training (70%) and testing (30%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# %% Check size of test set
len(X_test)  # Should return 9 (30% of 30 data points)

# %% Import LinearRegression model from scikit-learn
from sklearn.linear_model import LinearRegression

# %% Create an instance of the Linear Regression model
model = LinearRegression()

# %% Train (fit) the model using training data
model.fit(X_train, Y_train)

# %% Display actual test set salaries (Y_test)
Y_test

# %% Predict salaries for test set using trained model
y_pred = model.predict(X_test)

# %% Import mean squared error metric for evaluating model performance
from sklearn.metrics import mean_squared_error

# %% Compute and display Mean Squared Error (MSE)
mean_squared_error(Y_test, y_pred)  # Lower values indicate better model fit

# %% Compute and display model score (R² score) on test data
model.score(X_test, Y_test)  # R² value indicates how well the model explains the variance in data

# %% Import r2_score function for evaluating model performance
from sklearn.metrics import r2_score

# %% Compute and display R² score explicitly
r2_score(Y_test, y_pred)  # Should match the previous model.score() result

# %% Predict salary for 5 years of experience
model.predict([[5]])  # Outputs estimated salary for 5 years of experience

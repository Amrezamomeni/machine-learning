# Simple Linear Regression for student_scores.csv
# Google Colab ready

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Upload file in Colab
from google.colab import files
uploaded = files.upload()

# 2) Read CSV
df = pd.read_csv("student_scores.csv")

# 3) Quick inspection
print(df.head())
print(df.info())
print(df.describe())

# 4) Split features and target
X = df[['Hours']]
y = df['Scores']

# 5) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7) Predictions
y_pred = model.predict(X_test)

# 8) Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# 9) Regression equation
print(f"Regression Equation: Scores = {model.intercept_:.4f} + {model.coef_[0]:.4f} * Hours")

# 10) Predict a custom value
hours_value = np.array([[9.25]])
predicted_score = model.predict(hours_value)
print(f"Predicted score for 9.25 hours: {predicted_score[0]:.2f}")

# 11) Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores - Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

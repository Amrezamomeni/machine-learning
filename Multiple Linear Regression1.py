# ==========================================
# Multiple Linear Regression on Data.csv
# Columns: Hours, Practice, TeamWork, Scores
# ==========================================

# 1) Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2) Upload file in Google Colab
from google.colab import files
uploaded = files.upload()

# 3) Read dataset
df = pd.read_csv('Data.csv')

# 4) Show first rows
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# 5) Define features (X) and target (y)
X = df[['Hours', 'Practice', 'TeamWork']]
y = df['Scores']

# 6) Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7) Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 8) Predict
y_pred = model.predict(X_test)

# 9) Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# 10) Model coefficients
print("\nModel coefficients:")
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coeff_df)

print(f"\nIntercept: {model.intercept_:.4f}")

# Regression equation
print("\nRegression Equation:")
print(f"Scores = {model.intercept_:.4f} + "
      f"({model.coef_[0]:.4f} * Hours) + "
      f"({model.coef_[1]:.4f} * Practice) + "
      f"({model.coef_[2]:.4f} * TeamWork)")

# 11) Compare actual vs predicted
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print("\nActual vs Predicted:")
print(results)

# 12) Plot Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', s=80)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.grid(True)
plt.show()

# 13) Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred, y=residuals, s=80)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Scores")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# 14) Predict for a new sample
new_data = pd.DataFrame({
    'Hours': [6],
    'Practice': [3],
    'TeamWork': [2]
})

new_prediction = model.predict(new_data)
print("\nPrediction for new sample [Hours=6, Practice=3, TeamWork=2]:")
print(f"Predicted Score: {new_prediction[0]:.2f}")

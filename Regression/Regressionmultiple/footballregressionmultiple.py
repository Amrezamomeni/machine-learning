# ==========================================
# ۱. بارگذاری کتابخانه‌ها و فایل داده‌ها
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

# باز کردن پنجره آپلود فایل در گوگل کولب
try:
    from google.colab import files
    print("لطفاً فایل CSV فوتبال را آپلود کنید:")
    uploaded = files.upload()
    # خواندن اولین فایل آپلود شده
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(filename)
except Exception as e:
    # در صورت اجرا به صورت محلی
    print("فایل را به صورت محلی بارگذاری می‌کند...")
    df = pd.read_csv('football_dataset.csv')

# نمایش سطرها برای اطمینان از صحت بارگذاری
print("\n--- ۵ سطر اول داده‌های فوتبال ---")
print(df.head())

# ==========================================
# ۲. تعریف متغیرهای مستقل (ویژگی‌ها) و وابسته (هدف)
# ==========================================
# ویژگی‌ها برای پیش‌بینی تعداد گل‌ها
features = ['Matches_Played', 'Possession_Avg', 'Shots_on_Target', 'Pass_Accuracy']
target = 'Goals_Scored'

X = df[features]
y = df[target]

# تقسیم داده‌ها به بخش آموزش (Train) و آزمون (Test) - ۲۰ درصد داده‌ها برای آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# ۳. ساخت و آموزش مدل رگرسیون با Scikit-Learn
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# پیش‌بینی مقادیر مجموعه تست
y_pred = model.predict(X_test)

# نمایش فرمول و ضرایب خطی مدل
print("\n=== فرمول رگرسیون خطی به دست آمده ===")
equation = f"{target} = {model.intercept_:.4f}"
for col, coef in zip(features, model.coef_):
    equation += f" + ({coef:.4f} * {col})"
print(equation)

# ارزیابی معیارهای خطای مدل
mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

print("\n=== معیارهای ارزیابی مدل (تست) ===")
print(f"Mean Absolute Error (MAE)  : {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2 Score)       : {r2:.4f}")

# ==========================================
# ۴. تحلیل دقیق آماری (p-value ضرایب) با Statsmodels
# ==========================================
print("\n=== تحلیل آماری پیشرفته (OLS Summary) ===")
X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()
print(ols_model.summary())

# ==========================================
# ۵. رسم نمودار مقایسه گل‌های واقعی و پیش‌بینی‌شده
# ==========================================
plt.figure(figsize=(8, 6), dpi=100)
plt.scatter(y_test, y_pred, color='forestgreen', s=120, edgecolors='black', label='Test Samples', zorder=5)

# رسم خط پیش‌بینی کامل (y = x)
min_val = min(y.min(), y_pred.min()) - 1
max_val = max(y.max(), y_pred.max()) + 1
plt.plot([min_val, max_val], [min_val, max_val], color='darkred', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.title('Actual vs Predicted Goals Scored', fontsize=14, pad=15)
plt.xlabel('Actual Goals', fontsize=12)
plt.ylabel('Predicted Goals', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# ۱. خواندن داده‌ها (با فرض اینکه فایل آپلود شده است)
df = pd.read_csv("student_scores.csv")

# ۲. تعریف متغیرها
X = df[['Hours']]
y = df['Scores']

# ۳. تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ۴. ساخت و آموزش مدل رگرسیون خطی
model = LinearRegression()
model.fit(X_train, y_train)

# ۵. پیش‌بینی روی داده‌های تست
y_pred = model.predict(X_test)

# ۶. ارزیابی مدل با شاخص‌های مختلف (از جمله MAPE)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("--- شاخص‌های ارزیابی مدل سکولارن ---")
print(f"Intercept (عرض از مبدأ): {model.intercept_:.4f}")
print(f"Coefficient (ضریب زاویه خط): {model.coef_[0]:.4f}")
print(f"R2 Score (ضریب تعیین): {r2:.4f}")
print(f"MAE (میانگین خطای مطلق): {mae:.4f}")
print(f"RMSE (جذر میانگین مربعات خطا): {rmse:.4f}")
print(f"MAPE (میانگین درصد خطای مطلق): {mape:.2f}%")

# ۷. پیش‌بینی مقدار جدید بدون هشدار (با استفاده از DataFrame برای حفظ نام ستون)
custom_hour = pd.DataFrame([[9.25]], columns=['Hours'])
predicted_score = model.predict(custom_hour)
print(f"\nPredicted score for 9.25 hours: {predicted_score[0]:.2f}")

# ۸. تحلیل آماری پیشرفته‌تر و فواصل اطمینان با استفاده از Statsmodels
print("\n--- تحلیل آماری دقیق با Statsmodels ---")
X_with_const = sm.add_constant(X)  # اضافه کردن عرض از مبدأ به صورت دستی برای statsmodels
ols_model = sm.OLS(y, X_with_const).fit()
print(ols_model.summary())

# ۹. رسم نمودارها
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# نمودار اول: خط رگرسیون روی کل داده‌ها
sns.regplot(x='Hours', y='Scores', data=df, ax=axes[0], color='blue', line_kws={"color": "red"})
axes[0].set_title('Regression Line with 95% Confidence Interval')
axes[0].set_xlabel('Hours of Study')
axes[0].set_ylabel('Scores')
axes[0].grid(True)

# نمودار دوم: تحلیل باقیمانده‌ها (Residuals Plot) برای بررسی مفروضات رگرسیون
residuals = ols_model.resid
sns.scatterplot(x=ols_model.fittedvalues, y=residuals, ax=axes[1], color='purple')
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_title('Residuals vs Fitted Values')
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residuals')
axes[1].grid(True)

plt.tight_layout()
plt.show()

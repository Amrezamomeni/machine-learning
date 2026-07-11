import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.colab import files

# ۱. مرحله آپلود فایل
print("لطفاً فایل CSV خود را انتخاب کنید...")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# ۲. بارگذاری خودکار و تشخیص ستون‌ها
data = pd.read_csv(file_name)

# حذف ستون‌های اضافی احتمالی (مثل Unnamed: 0)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# انتخاب خودکار: ستون اول ورودی (X) و ستون دوم هدف (y)
feature_col = data.columns[0]
target_col = data.columns[1]

print(f"\n✅ فایل با موفقیت تحلیل شد:")
print(f"   - نام فایل: {file_name}")
print(f"   - متغیر مستقل (X): {feature_col}")
print(f"   - متغیر هدف (y): {target_col}")
print(f"   - تعداد کل داده‌ها: {data.shape[0]}")

# ۳. پیش‌پردازش و تقسیم داده‌ها
X = data[[feature_col]]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ۴. آموزش مدل
model = LinearRegression()
model.fit(X_train, y_train)
print("\n🚀 مدل با موفقیت آموزش دید.")

# ۵. ارزیابی فنی
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "="*35)
print("📊 گزارش ارزیابی فنی:")
print(f"دقت تبیین (R2): {r2:.4f}")
print(f"میانگین خطا (MAE): {mae:.4f}")
print(f"ریشه خطا (RMSE): {rmse:.4f}")
print("="*35)

# ۶. ترسیم نمودار هوشمند
plt.figure(figsize=(10, 6))
sns.regplot(x=feature_col, y=target_col, data=data, 
            scatter_kws={'color':'blue', 'alpha':0.5}, 
            line_kws={'color':'red'})
plt.title(f'Analysis: {feature_col} vs {target_col}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ۷. تست پیش‌بینی (به عنوان مثال برای ۱۰ واحد ورودی)
test_val = 10 
# اگر فایل دانش‌آموز باشد ۹.۲۵ را تست کنید، اگر تبلیغات باشد ۱۰۰ را.
prediction = model.predict([[test_val]])[0]
print(f"\n🎯 پیش‌بینی برای مقدار {test_val} در ستون {feature_col}: {prediction:.2f}")

# ۸. ذخیره مدل
model_save_name = "trained_model.joblib"
joblib.dump(model, model_save_name)
print(f"💾 مدل در فایل {model_save_name} ذخیره شد.")

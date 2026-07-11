# =============================================================================
# پروژه تحلیل رگرسیون پیشرفته و پیش‌بینی عملکرد تحصیلی
# طراح: امیررضا مومنی - تحلیلگر داده و پژوهشگر مدیریت دولتی
# متدولوژی: Linear Regression (OLS) با رویکرد ارزیابی چندگانه
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class StudentPerformancePredictor:
    """
    سیستم خبره پیش‌بینی نمرات مبتنی بر رگرسیون خطی.
    این کلاس شامل مراحل بارگذاری، پیش‌پردازش، آموزش، ارزیابی و ذخیره‌سازی مدل است.
    """
    
    def __init__(self, file_path):
        # مقداردهی اولیه و بارگذاری دیتاست
        self.file_path = file_path
        self.model = LinearRegression()
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        
    def load_and_preprocess(self):
        # بارگذاری داده‌ها و جداسازی متغیرهای مستقل (Features) و وابسته (Target)
        try:
            self.data = pd.read_csv(self.file_path)
            X = self.data[['Hours']]
            y = self.data['Scores']
            
            # تقسیم‌بندی داده‌ها به دو بخش آموزش (80%) و آزمون (20%) 
            # استفاده از random_state برای تکرارپذیری نتایج در تحقیقات علمی
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"✅ داده‌ها با موفقیت بارگذاری شدند. تعداد نمونه آموزش: {len(self.X_train)} | تست: {len(self.X_test)}")
        except Exception as e:
            print(f"❌ خطا در بارگذاری فایل: {e}")

    def train_model(self):
        # اجرای الگوریتم یادگیری نظارت‌شده (Supervised Learning)
        self.model.fit(self.X_train, self.y_train)
        print("🚀 مدل رگرسیون خطی با موفقیت آموزش دید.")

    def evaluate(self):
        # ارزیابی عملکرد مدل با استفاده از شاخص‌های آماری استاندارد
        predictions = self.model.predict(self.X_test)
        
        metrics = {
            "R2 Score (دقت تبیین)": r2_score(self.y_test, predictions),
            "MAE (میانگین خطای مطلق)": mean_absolute_error(self.y_test, predictions),
            "RMSE (ریشه میانگین مربعات خطا)": np.sqrt(mean_squared_error(self.y_test, predictions)),
            "MAPE (درصد خطای مطلق)": np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100
        }
        
        print("\n" + "="*30)
        print("📊 گزارش ارزیابی فنی مدل:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("="*30)
        return metrics

    def visualize(self):
        # ترسیم گرافیکی خط رگرسیون و توزیع داده‌ها جهت تحلیل بصری
        plt.figure(figsize=(10, 6))
        sns.regplot(x='Hours', y='Scores', data=self.data, 
                    scatter_kws={'color':'blue', 's':50}, 
                    line_kws={'color':'red', 'linewidth':2}, label='Best Fit Line')
        
        plt.title('Academic Performance Analysis: Hours vs Scores', fontsize=14)
        plt.xlabel('Study Hours', fontsize=12)
        plt.ylabel('Exam Scores', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def save_artifacts(self, model_name="academic_model.joblib"):
        # سریال‌سازی مدل برای استفاده در محیط Production یا وب‌سایت انجمن علمی
        joblib.dump(self.model, model_name)
        print(f"💾 مدل با نام '{model_name}' ذخیره شد.")

# -----------------------------------------------------------------------------
# اجرای خط لوله (Pipeline) اصلی برنامه
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. آدرس فایل دیتاست (در گوگل کولب همین نام کافی است)
    PATH = "student_scores.csv"
    
    # 2. ایجاد نمونه از کلاس
    predictor = StudentPerformancePredictor(PATH)
    
    # 3. اجرای مراحل به ترتیب منطقی
    predictor.load_and_preprocess()
    predictor.train_model()
    predictor.evaluate()
    predictor.visualize()
    
    # 4. ذخیره‌سازی نهایی
    predictor.save_artifacts()

    # 5. تست عملیاتی مدل: پیش‌بینی برای 9.25 ساعت مطالعه
    demo_input = pd.DataFrame([[9.25]], columns=['Hours'])
    result = predictor.model.predict(demo_input)
    print(f"\n🎯 پیش‌بینی نهایی برای ۹.۲۵ ساعت مطالعه: {result[0]:.2f}")

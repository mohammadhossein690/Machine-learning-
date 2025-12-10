# هایپرپارامترها در یادگیری ماشین

## تفاوت پارامتر و هایپرپارامتر

### پارامتر (Parameter)
- **تعریف**: مقادیری که مدل در طول فرآیند آموزش **یاد می‌گیرد**
- **مثال‌ها**:
  - ضرایب در رگرسیون خطی
  - وزن‌ها در شبکه‌های عصبی
  - بایاس در مدل‌های مختلف
- **ویژگی‌ها**:
  - توسط الگوریتم بهینه‌سازی تنظیم می‌شوند
  - از داده‌های آموزشی استخراج می‌شوند
  - ذخیره شده و برای پیش‌بینی استفاده می‌شوند

### هایپرپارامتر (Hyperparameter)
- **تعریف**: پارامترهایی که **قبل از آموزش** مدل تنظیم می‌شوند
- **مثال‌ها**:
  - نرخ یادگیری (Learning Rate)
  - تعداد لایه‌ها در شبکه عصبی
  - تعداد درختان در جنگل تصادفی
  - عمق درخت تصمیم
- **ویژگی‌ها**:
  - توسط داده‌ساز/مهندس ML تنظیم می‌شوند
  - بر فرآیند آموزش تأثیر می‌گذارند
  - بر اساس تجربه یا جستجوی سیستماتیک انتخاب می‌شوند

## هایپرپارامترهای رایج در مدل‌های مختلف

### 1. **مدل‌های مبتنی بر درخت**
- `max_depth`: حداکثر عمق درخت
- `min_samples_split`: حداقل نمونه برای تقسیم گره
- `min_samples_leaf`: حداقل نمونه در هر برگ
- `n_estimators`: تعداد درختان (در مدل‌های ensemble)

### 2. **شبکه‌های عصبی**
- `learning_rate`: نرخ یادگیری
- `batch_size`: حجم دسته‌های آموزشی
- `epochs`: تعداد دوره‌های آموزش
- `hidden_layers`: تعداد لایه‌های پنهان
- `dropout_rate`: نرخ حذف واحدها

### 3. **ماشین بردار پشتیبان (SVM)**
- `C`: پارامتر تنظیم (Regularization)
- `gamma`: پارامتر هسته RBF
- `kernel`: نوع هسته (linear, rbf, poly)

### 4. **رگرسیون لجستیک**
- `C`: قدرت regularization
- `penalty`: نوع جریمه (l1, l2)

## روش‌های انتخاب هایپرپارامترها

### 1. **جستجوی شبکه‌ای (Grid Search)**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**مزایا**:
- بررسی کامل فضای پارامترها
- موازی‌سازی آسان

**معایب**:
- هزینه محاسباتی بالا
- رشد نمایی با افزایش پارامترها

### 2. **جستجوی تصادفی (Random Search)**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(estimator, param_dist, n_iter=100, cv=5)
```

**مزایا**:
- کارآمدتر برای فضای پارامترهای بزرگ
- کشف نقاط بهینه غیرمنتظره

### 3. **بهینه‌سازی بیزی (Bayesian Optimization)**
```python
from skopt import BayesSearchCV

opt = BayesSearchCV(
    estimator,
    {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'max_depth': (1, 50)
    },
    n_iter=50,
    cv=3
)
```

**مزایا**:
- استفاده از اطلاعات تکرارهای قبلی
- نیاز به تعداد تکرار کمتر

### 4. **الگوریتم‌های تکاملی**
- الگوریتم ژنتیک
- بهینه‌سازی ازدحام ذرات (PSO)

### 5. **روش‌های مبتنی بر Gradient**
- فقط برای هایپرپارامترهای پیوسته کاربرد دارد
- نیاز به مشتق‌پذیری دارد

## بهترین روش‌های انتخاب هایپرپارامتر

### **راهنمای گام به گام**:

1. **تعریف فضای جستجو**:
   - محدوده‌های معقول برای هر هایپرپارامتر
   - استفاده از دانش دامنه مسئله

2. **انتخاب روش بهینه‌سازی**:
   - فضای کوچک: Grid Search
   - فضای بزرگ: Random Search یا Bayesian Optimization
   - مدل‌های پیچیده: روش‌های پیشرفته‌تر

3. **اعتبارسنجی متقابل**:
   ```python
   from sklearn.model_selection import cross_val_score
   
   scores = cross_val_score(model, X, y, cv=5)
   ```

4. **ارزیابی روی داده Validation**:
   - عدم استفاده از داده تست برای تنظیم
   - جداسازی داده Validation از Training

## نکات مهم و بهترین روش‌ها

### 1. **جلوگیری از بیش‌برازش (Overfitting)**:
- از Regularization مناسب استفاده کنید
- هایپرپارامترها را روی داده Validation تنظیم کنید
- از Early Stopping استفاده کنید

### 2. **مدیریت منابع محاسباتی**:
- شروع با مقادیر پیش‌فرض
- استفاده از یادگیری انتقالی (Transfer Learning)
- بهره‌گیری از محاسبات توزیع‌شده

### 3. **روندهای جدید**:
- **AutoML**: سیستم‌های خودکار تنظیم هایپرپارامتر
- **Neural Architecture Search**: جستجوی خودکار معماری شبکه
- **Hyperparameter Tuning as a Service**: سرویس‌های ابری

## مثال عملی کامل

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint

# بارگیری داده‌ها
data = load_iris()
X, y = data.data, data.target

# تقسیم داده‌ها
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف مدل
model = RandomForestClassifier(random_state=42)

# تعریف فضای پارامترها
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

# جستجوی تصادفی
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# آموزش و تنظیم
random_search.fit(X_train, y_train)

# نتایج
print("بهترین پارامترها:", random_search.best_params_)
print("بهترین امتیاز:", random_search.best_score_)
```

## جمع‌بندی

| جنبه | پارامتر | هایپرپارامتر |
|------|---------|--------------|
| **تعیین کننده** | مدل | انسان/الگوریتم |
| **زمان تنظیم** | حین آموزش | قبل از آموزش |
| **وابستگی به داده** | وابسته | مستقل |
| **مثال** | وزن‌های شبکه عصبی | نرخ یادگیری |

**توصیه نهایی**: 
- همیشه با مقادیر پیش‌فرض شروع کنید
- از اعتبارسنجی متقابل استفاده کنید
- داده تست را برای ارزیابی نهایی نگه دارید
- تعادل بین دقت و پیچیدگی مدل را حفظ کنید

این فایل می‌تواند در گیتهاب با نام `hyperparameters-guide.md` ذخیره شود.

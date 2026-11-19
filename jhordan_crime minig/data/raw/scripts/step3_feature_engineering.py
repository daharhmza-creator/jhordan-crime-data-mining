import os
import pandas as pd

# 1. مسارات الملفات
input_path = "data/processed/psd_opendata_features_selected.csv"
raw_cleaned_path = "data/processed/psd_opendata_cleaned.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "psd_opendata_engineered.csv")

# قراءة الملف الناتج من خطوة الترميز والقياس واختيار الميزات
df = pd.read_csv(input_path)
print("Input shape before feature engineering:", df.shape)

# 2. تجهيز مرجع زمني كـ Series
date_col = None
for col in ["timestamp", "datetime", "date"]:
    if col in df.columns:
        date_col = col
        break

if date_col is not None:
    dt = pd.Series(pd.to_datetime(df[date_col], errors="coerce"), index=df.index)
else:
    if os.path.exists(raw_cleaned_path):
        df_clean = pd.read_csv(raw_cleaned_path)
        years = pd.to_numeric(df_clean.get("year", 2024), errors="coerce").fillna(2024).astype(int)
    else:
        years = pd.Series([2024] * len(df))

    dt = pd.Series(
        pd.to_datetime(
            [
                f"{int(y)}-{((i % 12) + 1):02d}-{((i * 5 % 28) + 1):02d} {((i * 6) % 24):02d}:00:00"
                for i, y in enumerate(years[: len(df)])
            ]
        ),
        index=df.index,
    )

# 3. تقسيم الأوقات إلى فترات زمنية (Time Windows)
bins = [-1, 5, 11, 17, 23]
labels = ["Night (00-05)", "Morning (06-11)", "Afternoon (12-17)", "Evening (18-23)"]
df["time_window"] = pd.cut(dt.dt.hour, bins=bins, labels=labels).astype(str)

# ترميز فترات الوقت رقمياً (One-Hot) لسهولة استخدامها في النماذج
time_dummies = pd.get_dummies(df["time_window"], prefix="win", dtype=int)
df = pd.concat([df, time_dummies], axis=1)

# 4. أعمدة ثنائية (Binary Flags) لعطلة نهاية الأسبوع، أيام العمل، والعطل الرسمية
# في الأردن عطلة نهاية الأسبوع الجمعة والسبت (dayofweek: 4=Fri, 5=Sat)
is_weekend = dt.dt.dayofweek.isin([4, 5]).astype(int)
df["is_weekend"] = is_weekend
df["is_workday"] = (1 - is_weekend).astype(int)

# أمثلة عطل وطنية ثابتة (رأس السنة، عيد العمال، عيد الاستقلال، عيد الميلاد)
holidays_md = {(1, 1), (5, 1), (5, 25), (12, 25)}
df["is_national_holiday"] = [
    1 if (m, d) in holidays_md else 0 for m, d in zip(dt.dt.month, dt.dt.day)
]

print("Shape after feature engineering:", df.shape)
print("\nEngineered columns preview:")
preview_cols = ["time_window", "is_weekend", "is_workday", "is_national_holiday"]
print(df[preview_cols].head())

# 5. حفظ النتيجة النهائية
df.to_csv(output_path, index=False)
print("\nSaved engineered dataset to:", output_path)
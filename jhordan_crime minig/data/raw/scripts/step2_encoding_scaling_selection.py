import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 1. مسارات الملفات
input_path = "data/processed/psd_opendata_cleaned.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "psd_opendata_features_selected.csv")

# قراءة البيانات النظيفة
df = pd.read_csv(input_path)
print("Original cleaned shape:", df.shape)

# ---------------------------------------------------------
# النقطة 3: Encoding & Scaling
# ---------------------------------------------------------
categorical_cols = ["governorate", "district", "crime_category"]

encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_array = encoder.fit_transform(df[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)
df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

numeric_spatial_cols = ["year", "crime_count", "latitude", "longitude"]

scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df[numeric_spatial_cols])
df_scaled = pd.DataFrame(scaled_array, columns=numeric_spatial_cols, index=df.index)

df_processed = pd.concat([df_scaled, df_encoded], axis=1)

# ---------------------------------------------------------
# النقطة 4: Feature Selection باستخدام Correlation Matrix
# ---------------------------------------------------------
corr_matrix = df_processed.corr().abs()

print("\n--- Correlation Matrix (Top correlations with crime_count) ---")
if "crime_count" in corr_matrix.columns:
    print(corr_matrix["crime_count"].sort_values(ascending=False))

# استخراج المثلث العلوي فوق القطر الرئيسي بدون أخطاء أبعاد
upper_mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
upper_tri = corr_matrix.where(upper_mask)

threshold = 0.95
to_drop = [col for col in upper_tri.columns if upper_tri[col].gt(threshold).any()]

print("\nRedundant features dropped (corr >", threshold, "):", to_drop)
df_selected = df_processed.drop(columns=to_drop)

print("Final shape after feature selection:", df_selected.shape)

# حفظ النتيجة النهائية
df_selected.to_csv(output_path, index=False)
print("Saved final feature-selected dataset to:", output_path)

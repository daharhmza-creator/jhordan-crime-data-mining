import os
import pandas as pd

input_path = "data/raw/psd_opendata_raw.csv"
output_dir = "data/processed"
output_path = os.path.join(output_dir, "psd_opendata_cleaned.csv")

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)
print("Reading from:", os.path.abspath(input_path))
print(df.head(2))
print("Records before cleaning:", len(df))

df["district"] = df["district"].fillna("").astype(str).str.strip()
invalid_districts = ["", "nan", "NaN", "None", "null", "Unknown", "N/A", "-"]
df_clean = df[~df["district"].isin(invalid_districts)].copy()

df_clean["latitude"] = pd.to_numeric(df_clean["latitude"], errors="coerce")
df_clean["longitude"] = pd.to_numeric(df_clean["longitude"], errors="coerce")
df_clean = df_clean.dropna(subset=["latitude", "longitude"])

df_clean = df_clean[
    (df_clean["latitude"].between(29.0, 33.5))
    & (df_clean["longitude"].between(34.5, 39.5))
]

df_clean = df_clean.drop_duplicates().reset_index(drop=True)
df_clean.to_csv(output_path, index=False)

print("Records after cleaning:", len(df_clean))
print("Removed records:", len(df) - len(df_clean))
print("Saved cleaned file to:", output_path)

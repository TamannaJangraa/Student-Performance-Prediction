
# =====================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# STEP 1: UCI DATASET INSPECTION
# =====================================================

import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset location
DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"

print("\n======================================")
print(" UCI STUDENT PERFORMANCE DATASET")
print("======================================")

# Check dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

# Load dataset (semicolon separated)
df = pd.read_csv(DATA_PATH, sep=";")

# 1. Dataset dimensions
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\n2. COLUMN NAMES")
print(df.columns.tolist())

# 3. First five records
print("\n3. FIRST FIVE RECORDS")
print(df.head())

# 4. Dataset information
print("\n4. DATASET INFORMATION")
df.info()

# 5. Statistical summary
print("\n5. STATISTICAL SUMMARY")
print(df.describe().T)

# 6. Missing values
print("\n6. MISSING VALUES")
print(df.isnull().sum())

# 7. Duplicate records
print("\n7. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 8. Target variable
print("\n8. TARGET VARIABLE: G3")
print(df["G3"].describe())

# 9. Previous grades
print("\n9. PREVIOUS GRADES")
print(df[["G1", "G2", "G3"]].head())

# 10. Model input features
# Excluding G1, G2 and target G3
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

print("\n10. FINAL MODEL FEATURES")
print(X.columns.tolist())

print("\nTotal input features:", X.shape[1])
print("Target variable:", y.name)

print("\n======================================")
print(" DATASET INSPECTION COMPLETED!")
print("======================================")

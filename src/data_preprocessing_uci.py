
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# STEP 3: UCI DATA PREPROCESSING
# Dataset: UCI Student Performance - Mathematics
# Target: G3
# Excluded Features: G1, G2
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n========== LOADING UCI DATASET ==========")

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ------------------------------------------------------------
# 3. BASIC DATA VALIDATION
# ------------------------------------------------------------

print("\n========== DATA VALIDATION ==========")

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Records:", df.duplicated().sum())

print("\nDataset Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. DEFINE TARGET AND INPUT FEATURES
# ------------------------------------------------------------

print("\n========== FEATURE SELECTION ==========")

# G3 is the final Mathematics grade
y = df["G3"]

# Exclude G1 and G2 to predict final grade
# using information available before those assessments.

X = df.drop(columns=["G1", "G2", "G3"])

print("Input Features:")
print(X.columns.tolist())

print("\nTarget Variable: G3")

print("Total Input Features:", X.shape[1])


# ------------------------------------------------------------
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\n========== DATA SPLITTING ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# ------------------------------------------------------------
# 7. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ------------------------------------------------------------
# 8. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ------------------------------------------------------------
# 9. CREATE COLUMN TRANSFORMER
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ------------------------------------------------------------
# 10. FIT AND TRANSFORM DATA
# ------------------------------------------------------------

print("\n========== DATA PREPROCESSING ==========")

# Fit only on training data to prevent data leakage
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)

print("Training Data Shape:", X_train_processed.shape)
print("Testing Data Shape:", X_test_processed.shape)


# ------------------------------------------------------------
# 11. GET PROCESSED FEATURE NAMES
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

print("\nProcessed Feature Names:")
print(feature_names.tolist())

print("Total Processed Features:", len(feature_names))


# ------------------------------------------------------------
# 12. SAVE PREPROCESSOR
# ------------------------------------------------------------

joblib.dump(
    preprocessor,
    MODEL_DIR / "uci_preprocessor.pkl"
)


# ------------------------------------------------------------
# 13. SAVE PROCESSED DATA
# ------------------------------------------------------------


processed_data = {

    # Original data for leakage-free cross-validation
    "X_train_raw": X_train,
    "X_test_raw": X_test,

    # Already processed data for existing application
    "X_train": X_train_processed,
    "X_test": X_test_processed,

    "y_train": y_train,
    "y_test": y_test,

    "feature_names": feature_names,

    "numerical_features": numerical_features,
    "categorical_features": categorical_features

}

joblib.dump(
    processed_data,
    MODEL_DIR / "uci_processed_data.pkl"
)

print("Processed Data Saved Successfully!")


joblib.dump(
    processed_data,
    MODEL_DIR / "uci_processed_data.pkl"
)


# ------------------------------------------------------------
# 14. SAVE FEATURE INFORMATION
# ------------------------------------------------------------

feature_info = pd.DataFrame({

    "Feature": X.columns,

    "Data Type": X.dtypes.astype(str).values,

    "Feature Type": [
        "Numerical" if col in numerical_features
        else "Categorical"
        for col in X.columns
    ]

})

feature_info.to_csv(
    REPORT_DIR / "uci_feature_information.csv",
    index=False
)


# ------------------------------------------------------------
# 15. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========== PREPROCESSING SUMMARY ==========")

print("Original Dataset Shape:", df.shape)

print("Input Features:", X.shape[1])

print("Excluded Features: G1, G2")

print("Target Variable: G3")

print("Training Records:", X_train_processed.shape[0])

print("Testing Records:", X_test_processed.shape[0])

print("Processed Training Features:", X_train_processed.shape[1])

print("Processed Testing Features:", X_test_processed.shape[1])

print("\nPreprocessor Saved Successfully!")

print("Processed Data Saved Successfully!")

print("\n========== UCI PREPROCESSING COMPLETED ==========")


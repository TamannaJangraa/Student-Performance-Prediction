
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# STEP 4: UCI MACHINE LEARNING MODEL TRAINING
# Dataset: UCI Student Performance - Mathematics
# Target: G3
# Features: Excluding G1 and G2
# Algorithms: 6 Regression Models
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD PROCESSED DATA
# ------------------------------------------------------------

print("\n========== LOADING PROCESSED DATA ==========")

data = joblib.load(
    MODEL_DIR / "uci_processed_data.pkl"
)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("Training Records:", len(y_train))
print("Testing Records:", len(y_test))


# ------------------------------------------------------------
# 3. DEFINE SIX MACHINE LEARNING MODELS
# ------------------------------------------------------------

print("\n========== DEFINING SIX MODELS ==========")

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=10,
        epsilon=0.1,
        gamma="scale"
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )

}

print("Total Models:", len(models))

for model_name in models:
    print("-", model_name)


# ------------------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------

print("\n========== MODEL TRAINING STARTED ==========")

results = []
trained_models = {}
prediction_records = []

for name, model in models.items():

    print("\n--------------------------------------")
    print("Training:", name)
    print("--------------------------------------")

    # 5-Fold Cross Validation
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()

    # Train model on complete training data
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test, predictions
    )

    mse = mean_squared_error(
        y_test, predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test, predictions
    )

    # Store results
    results.append({

        "Model": name,
        "CV RMSE": cv_rmse,
        "CV RMSE Std": cv_std,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2

    })

    trained_models[name] = model

    # Store individual predictions
    for actual, predicted in zip(
        y_test, predictions
    ):

        prediction_records.append({

            "Model": name,
            "Actual Grade": actual,
            "Predicted Grade": predicted,
            "Residual": actual - predicted

        })

    print("CV RMSE:", round(cv_rmse, 4))
    print("CV Standard Deviation:", round(cv_std, 4))
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


# ------------------------------------------------------------
# 5. CREATE MODEL COMPARISON TABLE
# ------------------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="CV RMSE",
    ascending=True
).reset_index(drop=True)

comparison.insert(
    0,
    "Rank",
    range(1, len(comparison) + 1)
)

print("\nFINAL MODEL COMPARISON:\n")

print(
    comparison.to_string(index=False)
)


# ------------------------------------------------------------
# 6. SELECT BEST MODEL
# ------------------------------------------------------------

best_model_name = comparison.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_row = comparison.iloc[0]

print("\n========== BEST MODEL SELECTION ==========")

print("Best Model:", best_model_name)

print("CV RMSE:", round(best_row["CV RMSE"], 4))

print("Test MAE:", round(best_row["MAE"], 4))

print("Test MSE:", round(best_row["MSE"], 4))

print("Test RMSE:", round(best_row["RMSE"], 4))

print("Test R2 Score:", round(best_row["R2 Score"], 4))


# ------------------------------------------------------------
# 7. SAVE BEST MODEL
# ------------------------------------------------------------

joblib.dump(
    best_model,
    MODEL_DIR / "uci_best_model.pkl"
)

print("\nBest Model Saved Successfully!")


# ------------------------------------------------------------
# 8. SAVE MODEL COMPARISON
# ------------------------------------------------------------

comparison.to_csv(
    REPORT_DIR / "uci_model_comparison.csv",
    index=False
)

print("Model Comparison Saved Successfully!")


# ------------------------------------------------------------
# 9. SAVE PREDICTION RESULTS
# ------------------------------------------------------------

all_predictions = pd.DataFrame(
    prediction_records
)

all_predictions.to_csv(
    REPORT_DIR / "uci_prediction_results.csv",
    index=False
)

print("Prediction Results Saved Successfully!")


# ------------------------------------------------------------
# 10. SAVE BEST MODEL PREDICTIONS
# ------------------------------------------------------------

best_predictions = all_predictions[
    all_predictions["Model"] == best_model_name
].copy()

best_predictions.to_csv(
    REPORT_DIR / "uci_best_model_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 11. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("UCI MODEL TRAINING COMPLETED SUCCESSFULLY")
print("==========================================")

print("Dataset: UCI Student Mathematics")

print("Target Variable: G3")

print("Excluded Features: G1, G2")

print("Input Features: 30")

print("Algorithms Trained:", len(models))

print("Best Model:", best_model_name)

print("Best CV RMSE:", round(best_row["CV RMSE"], 4))

print("Best Test RMSE:", round(best_row["RMSE"], 4))

print("Best Test R2 Score:", round(best_row["R2 Score"], 4))

print("\nSAVED FILES:")

print("models/uci_best_model.pkl")

print("reports/uci_model_comparison.csv")

print("reports/uci_prediction_results.csv")

print("reports/uci_best_model_predictions.csv")

print("\n==========================================")

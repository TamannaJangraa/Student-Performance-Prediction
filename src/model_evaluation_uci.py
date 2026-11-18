
# ============================================================
# STEP 5: MODEL EVALUATION AND VISUALIZATION
# UCI STUDENT PERFORMANCE PREDICTION
# ============================================================

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
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

FIGURE_DIR = REPORT_DIR / "uci_evaluation"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 2. LOAD MODEL AND DATA
# ------------------------------------------------------------

print("\n========== LOADING MODEL ==========")

model = joblib.load(
    MODEL_DIR / "uci_best_model.pkl"
)

data = joblib.load(
    MODEL_DIR / "uci_processed_data.pkl"
)

comparison = pd.read_csv(
    REPORT_DIR / "uci_model_comparison.csv"
)

X_test = data["X_test"]
y_test = data["y_test"]

# ------------------------------------------------------------
# 3. GENERATE PREDICTIONS
# ------------------------------------------------------------

predictions = model.predict(X_test)

residuals = np.asarray(y_test) - predictions

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\n========== FINAL MODEL EVALUATION ==========")

print("Model:", type(model).__name__)
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# ------------------------------------------------------------
# 4. ACTUAL VS PREDICTED GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

plt.plot(
    [0, 20],
    [0, 20],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade (G3)")

plt.title("Actual vs Predicted Grades - Random Forest")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_actual_vs_predicted.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 5. RESIDUAL DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.histplot(
    residuals,
    kde=True
)

plt.axvline(0, linestyle="--")

plt.xlabel("Prediction Residuals")

plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_residual_distribution.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 6. RESIDUALS VS PREDICTED
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Grades")

plt.ylabel("Residuals")

plt.title("Residuals vs Predicted Values")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_residual_analysis.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 7. MODEL RMSE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="RMSE"
)

plt.title("RMSE Comparison of Regression Models")

plt.ylabel("RMSE - Lower is Better")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_rmse_comparison.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 8. R2 SCORE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.barplot(
    data=comparison,
    x="Model",
    y="R2 Score"
)

plt.title("R2 Score Comparison of Regression Models")

plt.ylabel("R2 Score - Higher is Better")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_r2_comparison.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------

if hasattr(model, "feature_importances_"):

    # Load fitted preprocessor
    preprocessor = joblib.load(
        MODEL_DIR / "uci_preprocessor.pkl"
    )

    # Extract transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Get feature importance values
    feature_values = model.feature_importances_

    # Verify dimensions
    if len(feature_names) != len(feature_values):

        raise ValueError(
            "Feature names and importance dimensions do not match!"
        )

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": feature_values

    })

    # Sort by importance
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    # Plot top 15 features
    plt.figure(figsize=(11, 8))

    sns.barplot(
        data=importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Top 15 Feature Importances - Random Forest"
    )

    plt.xlabel("Feature Importance")

    plt.ylabel("Input Features")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "06_feature_importance.png",
        dpi=300
    )

    plt.close()

    # Save complete feature importance
    importance.to_csv(
        REPORT_DIR / "uci_feature_importance.csv",
        index=False
    )

    print("\nFeature Importance Analysis Completed!")

else:

    print("\nSelected model does not support feature importance.")


# ------------------------------------------------------------
# 10. SAVE EVALUATION SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame([{

    "Model": type(model).__name__,

    "MAE": mae,

    "MSE": mse,

    "RMSE": rmse,

    "R2 Score": r2

}])

summary.to_csv(
    REPORT_DIR / "uci_evaluation_summary.csv",
    index=False
)

print("\n========== EVALUATION COMPLETED ==========")

print("All evaluation graphs saved in:")

print(FIGURE_DIR)

print("\nEvaluation summary saved successfully!")

print("==========================================")

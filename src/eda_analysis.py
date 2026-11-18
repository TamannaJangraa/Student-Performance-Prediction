
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: UCI Student Performance - Mathematics
# Target: G3
# G1 and G2 excluded from predictive analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "uci_eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH, sep=";")

# Exclude previous grades and target from input features
X = df.drop(columns=["G1", "G2", "G3"])

y = df["G3"]

print("\n==========================================")
print(" EXPLORATORY DATA ANALYSIS")
print("==========================================")

print("\nDataset Shape:", df.shape)

print("Input Features:", X.shape[1])

print("Target Variable:", y.name)

# ------------------------------------------------------------
# 3. DATA QUALITY ANALYSIS
# ------------------------------------------------------------

print("\n========== DATA QUALITY ==========")

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTarget Statistics:")
print(y.describe())

# Save descriptive statistics
df.describe(include="all").T.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)

# ------------------------------------------------------------
# 4. VISUALIZATION SETTINGS
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

plt.rcParams["figure.figsize"] = (10, 6)

plt.rcParams["font.size"] = 11


def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ------------------------------------------------------------
# 5. GRADE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="G3",
    bins=np.arange(-0.5, 21.5, 1),
    kde=True
)

plt.title("Distribution of Final Mathematics Grades")

plt.xlabel("Final Grade (G3)")

plt.ylabel("Number of Students")

plt.xticks(range(0, 21))

save_plot("01_grade_distribution.png")


# ------------------------------------------------------------
# 6. NUMERICAL CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_data = df.select_dtypes(
    include=np.number
).drop(columns=["G1", "G2"], errors="ignore")

correlation = numeric_data.corr()

plt.figure(figsize=(13, 10))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap - UCI Student Performance")

save_plot("02_correlation_heatmap.png")


# ------------------------------------------------------------
# 7. FEATURE CORRELATION WITH FINAL GRADE
# ------------------------------------------------------------

target_corr = correlation["G3"].drop("G3")

target_corr = target_corr.sort_values(
    key=abs,
    ascending=False
)

print("\n========== CORRELATION WITH G3 ==========")

print(target_corr)

plt.figure(figsize=(10, 7))

sns.barplot(
    x=target_corr.values,
    y=target_corr.index
)

plt.title("Numerical Feature Correlation with Final Grade")

plt.xlabel("Correlation Coefficient")

plt.ylabel("Features")

save_plot("03_feature_correlation.png")


# ------------------------------------------------------------
# 8. STUDY TIME VS FINAL GRADE
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="studytime",
    y="G3"
)

plt.title("Study Time vs Final Mathematics Grade")

plt.xlabel("Weekly Study Time Category")

plt.ylabel("Final Grade (G3)")

save_plot("04_studytime_vs_grade.png")


# ------------------------------------------------------------
# 9. PAST FAILURES VS FINAL GRADE
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="failures",
    y="G3"
)

plt.title("Past Failures vs Final Mathematics Grade")

plt.xlabel("Number of Past Class Failures")

plt.ylabel("Final Grade (G3)")

save_plot("05_failures_vs_grade.png")


# ------------------------------------------------------------
# 10. ABSENCES VS FINAL GRADE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="absences",
    y="G3",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"}
)

plt.title("Student Absences vs Final Mathematics Grade")

plt.xlabel("Number of Absences")

plt.ylabel("Final Grade (G3)")

save_plot("06_absences_vs_grade.png")


# ------------------------------------------------------------
# 11. CATEGORICAL FEATURE ANALYSIS
# ------------------------------------------------------------

categorical_columns = X.select_dtypes(
    include="object"
).columns.tolist()

print("\n========== CATEGORICAL FEATURES ==========")

print(categorical_columns)

# Average grade by gender
gender_analysis = df.groupby("sex")["G3"].agg(
    ["count", "mean", "median"]
)

print("\nGender-wise Performance:")
print(gender_analysis)

gender_analysis.to_csv(
    OUTPUT_DIR / "gender_analysis.csv"
)

# Average grade by school
school_analysis = df.groupby("school")["G3"].agg(
    ["count", "mean", "median"]
)

print("\nSchool-wise Performance:")
print(school_analysis)

school_analysis.to_csv(
    OUTPUT_DIR / "school_analysis.csv"
)


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print(" EDA COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nTotal Students:", len(df))

print("Input Features:", X.shape[1])

print("Target Variable:", y.name)

print("Average Final Grade:", round(y.mean(), 2))

print("Median Final Grade:", round(y.median(), 2))

print("\nEDA Reports Saved At:")

print(OUTPUT_DIR)

print("\nGenerated Visualizations:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file.name)

print("\n==========================================")


# ============================================================
# STUDENT MATHEMATICS PERFORMANCE PREDICTION
# PROFESSIONAL STREAMLIT WEB APPLICATION
# Dataset: UCI Student Performance - Mathematics
# Model: Random Forest Regressor
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "models" / "uci_best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "uci_preprocessor.pkl"

REPORT_DIR = BASE_DIR / "reports"

COMPARISON_PATH = REPORT_DIR / "uci_model_comparison.csv"
EVALUATION_PATH = REPORT_DIR / "uci_evaluation_summary.csv"
IMPORTANCE_PATH = REPORT_DIR / "uci_feature_importance.csv"
HISTORY_PATH = REPORT_DIR / "uci_prediction_history.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}

.hero {
    padding: 28px;
    border-radius: 16px;
    background: linear-gradient(120deg, #142D4E, #245B8F);
    color: white;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
    color: white;
}

.hero p {
    font-size: 15px;
    color: #DCEBFA;
    margin-bottom: 0;
}

.section-heading {
    font-size: 23px;
    font-weight: 650;
    margin-top: 12px;
    margin-bottom: 15px;
}

.result-card {
    padding: 30px;
    border-radius: 16px;
    background: linear-gradient(120deg, #174D42, #16785D);
    color: white;
    text-align: center;
}

.result-card h1 {
    font-size: 58px;
    color: white;
    margin: 8px;
}

.result-card p {
    color: #E0F2E9;
    font-size: 17px;
}

.small-note {
    color: #8A94A6;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. LOAD PROJECT ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    dataset = pd.read_csv(DATA_PATH, sep=";")

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return dataset, model, preprocessor


@st.cache_data
def load_csv(path_string):

    path = Path(path_string)

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


try:

    dataset, model, preprocessor = load_artifacts()

except Exception as error:

    st.error(f"Unable to load project files: {error}")

    st.info(
        "Check the dataset and model files inside "
        "the dataset and models folders."
    )

    st.stop()


# ============================================================
# 5. FEATURE INFORMATION
# ============================================================

TARGET = "G3"

EXCLUDED_COLUMNS = ["G1", "G2", "G3"]

# Use the same feature order used during model training.
INPUT_FEATURES = list(preprocessor.feature_names_in_)

comparison = load_csv(str(COMPARISON_PATH))
evaluation = load_csv(str(EVALUATION_PATH))
importance = load_csv(str(IMPORTANCE_PATH))


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def get_grade_status(grade):

    if grade >= 10:
        return "Passing Grade"

    return "Below Passing Grade"


def save_prediction(student_input, prediction):

    record = student_input.copy()

    record["Predicted G3"] = round(prediction, 2)

    record["Prediction Date"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if HISTORY_PATH.exists():

        old_history = pd.read_csv(HISTORY_PATH)

        updated_history = pd.concat(
            [old_history, record],
            ignore_index=True
        )

    else:

        updated_history = record

    updated_history.to_csv(
        HISTORY_PATH,
        index=False
    )

    return record


def show_hero(title, description):

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Student Performance AI")

    st.caption("ACADEMIC ANALYTICS PLATFORM")

    st.divider()

    selected_page = st.radio(
        "MAIN NAVIGATION",
        [
            "🏠 Dashboard",
            "🔮 Predict Performance",
            "📊 Data Analytics",
            "🤖 Model Performance",
            "📚 Prediction History",
            "ℹ️ About Project"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.success("Random Forest Regressor")

    st.caption("Dataset: UCI Mathematics")

    st.caption("Target: Final Grade (G3)")

    st.caption("Input Features: 30")

    st.caption("Excluded: G1 and G2")

    st.divider()

    st.caption("AIML Minor Project | 2026")


# ============================================================
# 8. DASHBOARD
# ============================================================

if selected_page == "🏠 Dashboard":

    show_hero(
        "Student Performance Dashboard",
        "Machine Learning Based Academic Performance Prediction System"
    )

    total_students = len(dataset)

    average_grade = dataset["G3"].mean()

    pass_percentage = (
        (dataset["G3"] >= 10).mean() * 100
    )

    total_features = len(INPUT_FEATURES)

    st.markdown("### 📌 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        f"{total_students:,}"
    )

    c2.metric(
        "Average Final Grade",
        f"{average_grade:.2f}/20"
    )

    c3.metric(
        "Passing Percentage",
        f"{pass_percentage:.1f}%"
    )

    c4.metric(
        "Input Features",
        total_features
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("### 📈 Grade Distribution")

        grade_counts = (
            dataset["G3"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            grade_counts,
            x_label="Final Grade (G3)",
            y_label="Number of Students"
        )

    with right:

        st.markdown("### 🎯 Academic Performance")

        performance = pd.DataFrame({

            "Category": [
                "Below Passing",
                "Passing"
            ],

            "Students": [
                (dataset["G3"] < 10).sum(),
                (dataset["G3"] >= 10).sum()
            ]

        })

        st.bar_chart(
            performance,
            x="Category",
            y="Students"
        )

    st.divider()

    st.markdown("### 📋 Dataset Information")

    info1, info2 = st.columns(2)

    with info1:

        st.write("**Dataset:** UCI Student Performance")

        st.write("**Subject:** Mathematics")

        st.write("**Target Variable:** G3")

        st.write("**Total Columns:**", dataset.shape[1])

    with info2:

        st.write("**Training Records:** 316")

        st.write("**Testing Records:** 79")

        st.write("**Model:** Random Forest")

        st.write("**Grade Scale:** 0 to 20")

    st.info(
        "The model predicts the final Mathematics grade "
        "using demographic, social and academic features. "
        "Previous grades G1 and G2 are excluded."
    )


# ============================================================
# 9. PREDICT PERFORMANCE
# ============================================================

elif selected_page == "🔮 Predict Performance":

    show_hero(
        "Student Grade Prediction",
        "Enter student information to estimate the final Mathematics grade."
    )

    st.info(
        "The prediction is based on 30 input features. "
        "G1 and G2 are excluded from this experiment."
    )

    input_data = {}

    with st.form("student_prediction_form"):

        st.markdown("### 👤 Student Information")

        col1, col2 = st.columns(2)

        for index, feature in enumerate(INPUT_FEATURES):

            current_col = col1 if index % 2 == 0 else col2

            with current_col:

                series = dataset[feature]

                label = feature.replace(
                    "_", " "
                ).title()

                if pd.api.types.is_numeric_dtype(series):

                    minimum = int(series.min())

                    maximum = int(series.max())

                    default = int(series.median())

                    input_data[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1,
                        key=f"input_{feature}"
                    )

                else:

                    options = sorted(
                        series.dropna().unique().tolist()
                    )

                    default = series.mode().iloc[0]

                    default_index = (
                        options.index(default)
                        if default in options else 0
                    )

                    input_data[feature] = st.selectbox(
                        label,
                        options,
                        index=default_index,
                        key=f"input_{feature}"
                    )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Final Grade",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        student_input = pd.DataFrame(
            [input_data],
            columns=INPUT_FEATURES
        )

        try:

            processed_input = preprocessor.transform(
                student_input
            )

            prediction = float(
                model.predict(processed_input)[0]
            )

            prediction = float(
                np.clip(prediction, 0, 20)
            )

            st.session_state["last_prediction"] = {
                "grade": prediction,
                "inputs": student_input
            }

            record = save_prediction(
                student_input,
                prediction
            )

            st.session_state["last_record"] = record

        except Exception as error:

            st.error(f"Prediction failed: {error}")

    # Display the latest prediction even after page reruns.

    if "last_prediction" in st.session_state:

        result = st.session_state["last_prediction"]

        prediction = result["grade"]

        student_input = result["inputs"]

        st.divider()

        st.markdown("### 🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <p>Predicted Final Mathematics Grade</p>
                <h1>{prediction:.2f} / 20</h1>
                <p>{get_grade_status(prediction)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(prediction / 20)

        if prediction >= 10:

            st.success(
                "The predicted grade is above the passing threshold."
            )

        else:

            st.warning(
                "The predicted grade is below the passing threshold."
            )

        st.markdown("### 📋 Student Input Summary")

        st.dataframe(
            student_input,
            use_container_width=True,
            hide_index=True
        )

        report = st.session_state["last_record"]

        st.download_button(
            "📥 Download Student Report",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_report.csv",
            mime="text/csv"
        )

        st.caption(
            "This prediction is an estimate generated by a "
            "machine learning model, not an official examination result."
        )


# ============================================================
# 10. DATA ANALYTICS
# ============================================================

elif selected_page == "📊 Data Analytics":

    show_hero(
        "Student Data Analytics",
        "Explore student demographics and academic performance patterns."
    )

    st.markdown("### 📊 Academic Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Study Time vs Final Grade")

        study_analysis = (
            dataset.groupby("studytime")["G3"]
            .mean()
            .reset_index()
        )

        study_analysis["Study Time"] = (
            study_analysis["studytime"].astype(str)
        )

        st.bar_chart(
            study_analysis,
            x="Study Time",
            y="G3"
        )

        st.caption(
            "Average final grade for each study-time category."
        )

    with col2:

        st.markdown("#### Failures vs Final Grade")

        failure_analysis = (
            dataset.groupby("failures")["G3"]
            .mean()
            .reset_index()
        )

        failure_analysis["Previous Failures"] = (
            failure_analysis["failures"].astype(str)
        )

        st.bar_chart(
            failure_analysis,
            x="Previous Failures",
            y="G3"
        )

        st.caption(
            "Average grade according to previous failures."
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        st.markdown("#### Gender-wise Performance")

        gender_analysis = (
            dataset.groupby("sex")["G3"]
            .mean()
            .reset_index()
        )

        st.bar_chart(
            gender_analysis,
            x="sex",
            y="G3"
        )

    with col4:

        st.markdown("#### Absences vs Final Grade")

        absence_analysis = (
            dataset.groupby("absences")["G3"]
            .mean()
            .reset_index()
        )

        st.line_chart(
            absence_analysis.set_index("absences")
        )

    st.divider()

    st.markdown("### 🔗 Correlation Analysis")

    numeric_data = dataset.select_dtypes(
        include=np.number
    )

    correlation = numeric_data.corr()

    st.dataframe(
        correlation.round(2),
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📑 Descriptive Statistics")

    st.dataframe(
        dataset.describe().round(2),
        use_container_width=True
    )


# ============================================================
# 11. MODEL PERFORMANCE
# ============================================================

elif selected_page == "🤖 Model Performance":

    show_hero(
        "Machine Learning Model Evaluation",
        "Compare regression algorithms and analyse the selected model."
    )

    if not comparison.empty:

        st.markdown("### 🏆 Model Comparison")

        display_comparison = comparison.copy()

        if "CV RMSE" in display_comparison.columns:

            display_comparison = display_comparison.sort_values(
                "CV RMSE",
                ascending=True
            )

        st.dataframe(
            display_comparison.round(4),
            use_container_width=True,
            hide_index=True
        )

        if "RMSE" in comparison.columns:

            st.markdown("### 📉 RMSE Comparison")

            rmse_data = comparison.set_index(
                "Model"
            )[["RMSE"]]

            st.bar_chart(rmse_data)

            st.caption(
                "Lower RMSE indicates lower prediction error."
            )

        if "R2 Score" in comparison.columns:

            st.markdown("### 📈 R² Score Comparison")

            r2_data = comparison.set_index(
                "Model"
            )[["R2 Score"]]

            st.bar_chart(r2_data)

            st.caption(
                "Higher R² indicates better explanatory fit."
            )

    else:

        st.warning(
            "Model comparison report is unavailable."
        )

    st.divider()

    st.markdown("### 🎯 Selected Model Evaluation")

    if not evaluation.empty:

        metrics = evaluation.iloc[0]

        a, b, c, d = st.columns(4)

        a.metric(
            "MAE",
            f"{metrics['MAE']:.4f}"
        )

        b.metric(
            "MSE",
            f"{metrics['MSE']:.4f}"
        )

        c.metric(
            "RMSE",
            f"{metrics['RMSE']:.4f}"
        )

        d.metric(
            "R² Score",
            f"{metrics['R2 Score']:.4f}"
        )

    else:

        st.info(
            "Evaluation summary is not available. "
            "Run model evaluation first."
        )

    st.divider()

    st.markdown("### 🔍 Feature Importance")

    if not importance.empty:

        top_features = importance.head(15)

        st.bar_chart(
            top_features.set_index("Feature")[["Importance"]]
        )

        st.dataframe(
            top_features.round(4),
            use_container_width=True,
            hide_index=True
        )

    elif hasattr(model, "feature_importances_"):

        feature_names = preprocessor.get_feature_names_out()

        importance_values = model.feature_importances_

        importance_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance_values

        }).sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.head(15).set_index("Feature")
        )

    else:

        st.info(
            "Feature importance is not available for this model."
        )

    st.warning(
        "R² is a regression metric, not classification accuracy. "
        "A lower error and higher R² are desirable."
    )


# ============================================================
# 12. PREDICTION HISTORY
# ============================================================

elif selected_page == "📚 Prediction History":

    show_hero(
        "Prediction History",
        "Review and download previously generated student predictions."
    )

    if HISTORY_PATH.exists():

        history = pd.read_csv(HISTORY_PATH)

        if not history.empty:

            total_predictions = len(history)

            average_prediction = history["Predicted G3"].mean()

            highest_prediction = history["Predicted G3"].max()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Predictions",
                total_predictions
            )

            c2.metric(
                "Average Predicted Grade",
                f"{average_prediction:.2f}"
            )

            c3.metric(
                "Highest Predicted Grade",
                f"{highest_prediction:.2f}"
            )

            st.divider()

            st.markdown("### 📋 Previous Records")

            st.dataframe(
                history.iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Complete History",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="uci_prediction_history.csv",
                mime="text/csv"
            )

            st.markdown("### 📊 Prediction Grade Distribution")

            st.bar_chart(
                history["Predicted G3"].value_counts(
                    bins=10
                ).sort_index()
            )

        else:

            st.info("No prediction records available.")

    else:

        st.info(
            "No prediction history available yet. "
            "Generate your first prediction to see records here."
        )


# ============================================================
# 13. ABOUT PROJECT
# ============================================================

elif selected_page == "ℹ️ About Project":

    show_hero(
        "About the Project",
        "An academic machine learning project for student performance prediction."
    )

    st.markdown("### 🎯 Project Objective")

    st.write(
        "The objective of this project is to develop a "
        "machine learning system that predicts students' "
        "final Mathematics grades using demographic, "
        "social and academic information."
    )

    st.markdown("### 📂 Dataset Information")

    st.write("**Dataset:** UCI Student Performance Dataset")

    st.write("**Subject:** Mathematics")

    st.write("**Records:** 395 students")

    st.write("**Original Features:** 33")

    st.write("**Input Features:** 30")

    st.write("**Target Variable:** G3")

    st.write("**Excluded Features:** G1 and G2")

    st.markdown("### ⚙️ Project Methodology")

    steps = [
        "Dataset Collection",
        "Data Exploration and Analysis",
        "Data Preprocessing",
        "Feature Selection",
        "Training and Testing Split",
        "Machine Learning Model Training",
        "Model Evaluation",
        "Final Grade Prediction"
    ]

    for index, step in enumerate(steps, start=1):

        st.write(f"**{index}.** {step}")

    st.markdown("### 🤖 Machine Learning Algorithms")

    algorithms = [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "Support Vector Regression",
        "Extra Trees Regressor"
    ]

    for algorithm in algorithms:

        st.write(f"• {algorithm}")

    st.markdown("### 🛠️ Technologies Used")

    technologies = [
        "Python",
        "Pandas and NumPy",
        "Scikit-learn",
        "Matplotlib and Seaborn",
        "Streamlit",
        "Joblib"
    ]

    st.write(" | ".join(technologies))

    st.divider()

    st.markdown("### 📌 Important Information")

    st.info(
        "G1 and G2 are excluded to evaluate the prediction "
        "of final grade G3 without using the first and second "
        "period grades as input."
    )

    st.caption(
        "This application is developed for academic and "
        "educational purposes."
    )


# ============================================================
# 14. GLOBAL FOOTER
# ============================================================

st.divider()

footer_left, footer_right = st.columns(2)

with footer_left:

    st.caption(
        "🎓 Student Mathematics Performance Prediction"
    )

with footer_right:

    st.caption(
        "UCI Dataset | AIML Minor Project | Python + Streamlit"
    )

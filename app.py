import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Loan Approval Prediction System")

st.markdown(
    """
    ### Machine Learning Based Loan Prediction

    Enter the applicant details below and the trained
    machine learning model will predict the loan status.
    """
)

st.divider()


# ============================================================
# PROJECT / MODEL PATH
# Works on both Windows and Streamlit Cloud
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


MODEL_PATH = MODEL_DIR / "best_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoders.pkl"
FEATURE_PATH = MODEL_DIR / "feature_names.pkl"


# ============================================================
# CHECK MODEL FILES
# ============================================================

missing_files = []

if not MODEL_PATH.exists():
    missing_files.append("model/best_model.pkl")

if not ENCODER_PATH.exists():
    missing_files.append("model/label_encoders.pkl")

if not FEATURE_PATH.exists():
    missing_files.append("model/feature_names.pkl")


if missing_files:

    st.error("❌ Required model files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Make sure the model folder and its files are uploaded "
        "to your GitHub repository."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_PATH)

    label_encoders = joblib.load(ENCODER_PATH)

    feature_names = joblib.load(FEATURE_PATH)

    return model, label_encoders, feature_names


# ============================================================
# LOAD MODEL FILES
# ============================================================

try:

    model, label_encoders, feature_names = load_model_files()

    st.success("✅ Machine Learning model loaded successfully!")

except Exception as e:

    st.error("❌ Error while loading the machine learning model.")

    st.exception(e)

    st.stop()


# ============================================================
# CONVERT FEATURE NAMES
# ============================================================

if isinstance(feature_names, np.ndarray):

    feature_names = feature_names.tolist()

elif isinstance(feature_names, tuple):

    feature_names = list(feature_names)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Loan Prediction")

st.sidebar.info(
    """
    This application uses a trained
    Machine Learning model to predict
    loan approval.
    """
)


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("👤 Applicant Information")


col1, col2 = st.columns(2)


# ============================================================
# PERSONAL INFORMATION
# ============================================================

with col1:

    st.subheader("Personal Details")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

with col2:

    st.subheader("Financial Details")

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=100
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    loan_amount_term = st.number_input(
        "Loan Amount Term",
        min_value=1,
        value=360,
        step=1
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict = st.button(
    "🔮 Predict Loan Approval",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    try:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "gender": [gender],

            "married": [married],

            "dependents": [dependents],

            "education": [education],

            "self_employed": [self_employed],

            "applicantincome": [applicant_income],

            "coapplicantincome": [coapplicant_income],

            "loanamount": [loan_amount],

            "loan_amount_term": [loan_amount_term],

            "credit_history": [credit_history],

            "property_area": [property_area]

        })


        # ----------------------------------------------------
        # ENCODE CATEGORICAL FEATURES
        # ----------------------------------------------------

        if isinstance(label_encoders, dict):

            for column, encoder in label_encoders.items():

                if column in input_data.columns:

                    try:

                        input_data[column] = encoder.transform(
                            input_data[column]
                        )

                    except ValueError:

                        st.error(
                            f"❌ Unknown value found in `{column}`."
                        )

                        st.stop()

        else:

            st.warning(
                "⚠️ Label encoders are not stored as a dictionary."
            )


        # ----------------------------------------------------
        # MATCH TRAINING FEATURES
        # ----------------------------------------------------

        input_data = input_data.reindex(
            columns=feature_names,
            fill_value=0
        )


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("📊 Prediction Result")


        # Convert prediction to string for safe comparison

        prediction_string = str(prediction).strip().lower()


        approved_values = [

            "1",

            "yes",

            "y",

            "approved",

            "loan approved",

            "true"

        ]


        if prediction_string in approved_values:

            st.success(
                "🎉 LOAN APPROVED"
            )

            st.markdown(
                """
                ### ✅ Congratulations!

                Based on the trained Machine Learning model,
                the applicant's loan is predicted to be **Approved**.
                """
            )

        else:

            st.error(
                "❌ LOAN NOT APPROVED"
            )

            st.markdown(
                """
                ### ❌ Loan Prediction

                Based on the trained Machine Learning model,
                the applicant's loan is predicted to be **Not Approved**.
                """
            )


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = float(
                np.max(probabilities) * 100
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


        # ----------------------------------------------------
        # SHOW INPUT DATA
        # ----------------------------------------------------

        st.subheader("📋 Applicant Details")


        applicant_details = pd.DataFrame({

            "Parameter": [

                "Gender",

                "Married",

                "Dependents",

                "Education",

                "Self Employed",

                "Applicant Income",

                "Co-applicant Income",

                "Loan Amount",

                "Loan Amount Term",

                "Credit History",

                "Property Area"

            ],

            "Value": [

                gender,

                married,

                dependents,

                education,

                self_employed,

                applicant_income,

                coapplicant_income,

                loan_amount,

                loan_amount_term,

                credit_history,

                property_area

            ]

        })


        st.dataframe(
            applicant_details,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.subheader("🤖 Model Information")

        st.write(
            "Prediction generated using the trained "
            "Machine Learning model."
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Prediction could not be completed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Loan Approval Prediction System | "
    "Machine Learning Project"
)
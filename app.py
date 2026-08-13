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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoders.pkl"
FEATURE_PATH = MODEL_DIR / "feature_names.pkl"


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Loan Approval Prediction System")

st.markdown(
    """
    ### Machine Learning Based Loan Prediction

    Enter the applicant details below to predict the loan status.
    """
)

st.divider()


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    MODEL_PATH,
    ENCODER_PATH,
    FEATURE_PATH
]

missing_files = [
    str(file.relative_to(BASE_DIR))
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("❌ Required model files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.code(file)

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


try:

    model, label_encoders, feature_names = load_model_files()

except Exception as e:

    st.error("❌ Failed to load model files.")

    st.exception(e)

    st.stop()


# ============================================================
# FEATURE NAMES
# ============================================================

if isinstance(feature_names, np.ndarray):
    feature_names = feature_names.tolist()

elif isinstance(feature_names, tuple):
    feature_names = list(feature_names)


# ============================================================
# MODEL STATUS
# ============================================================

st.success("✅ Machine Learning model loaded successfully!")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Loan Prediction")

st.sidebar.info(
    """
    This application predicts loan approval
    using a trained Machine Learning model.
    """
)

st.sidebar.write("Model: Best Model Pipeline")
st.sidebar.write("Classes: 0 = Rejected, 1 = Approved")


# ============================================================
# APPLICANT DETAILS
# ============================================================

st.header("👤 Applicant Information")

col1, col2 = st.columns(2)


# ============================================================
# PERSONAL DETAILS
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
# FINANCIAL DETAILS
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

predict_button = st.button(
    "🔮 Predict Loan Approval",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE RAW INPUT
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
        # IMPORTANT:
        # best_model.pkl is a PIPELINE.
        #
        # We first try the raw data because the pipeline
        # may already contain preprocessing.
        # ----------------------------------------------------

        input_data = input_data[feature_names]

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = float(
                np.max(probabilities) * 100
            )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("📊 Prediction Result")


        if int(prediction) == 1:

            st.success(
                "🎉 LOAN APPROVED"
            )

            st.markdown(
                """
                ### ✅ Loan Approved

                The Machine Learning model predicts that
                this application is likely to be approved.
                """
            )

        else:

            st.error(
                "❌ LOAN NOT APPROVED"
            )

            st.markdown(
                """
                ### ❌ Loan Not Approved

                The Machine Learning model predicts that
                this application is likely to be rejected.
                """
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence is not None:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


        # ----------------------------------------------------
        # INPUT SUMMARY
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


    except Exception as e:

        st.error(
            "❌ Prediction failed."
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
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)


# ==========================================
# LOAD MACHINE LEARNING FILES
# ==========================================

@st.cache_resource
def load_model_files():
    model = joblib.load("model/best_model.pkl")
    label_encoders = joblib.load("model/label_encoders.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return model, label_encoders, feature_names


@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_dataset.csv")


model, label_encoders, feature_names = load_model_files()
df = load_data()
model_results = pd.read_csv("model/model_results.csv")


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏦 Loan Approval System")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔮 Loan Prediction",
        "📊 Model Comparison",
        "📈 EDA Dashboard"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning based system for predicting "
    "loan approval or rejection."
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🏦 Loan Approval Prediction System")

    st.subheader(
        "Machine Learning Based Loan Decision Support System"
    )

    st.write("""
    This project uses Machine Learning to predict whether a loan
    application is likely to be approved or rejected.

    The system analyzes important applicant and financial information
    such as income, loan amount, credit history, education and
    property area.
    """)

    st.divider()

    best_model_row = model_results.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Records",
            f"{len(df)}"
        )

    with col2:
        st.metric(
            "Input Features",
            f"{len(feature_names)}"
        )

    with col3:
        st.metric(
            "ML Models Compared",
            f"{len(model_results)}"
        )

    with col4:
        st.metric(
            "Best Accuracy",
            f"{best_model_row['Accuracy']}%"
        )

    st.divider()

    st.subheader("🤖 Selected Best Model")

    st.success(
        f"🏆 {best_model_row['Model']} "
        f"achieved the highest accuracy of "
        f"{best_model_row['Accuracy']}%."
    )

    st.subheader("⚙️ System Workflow")

    st.write(
        """
        **1. Data Collection**
        → **2. Data Preprocessing**
        → **3. Feature Encoding**
        → **4. Model Training**
        → **5. Model Comparison**
        → **6. Best Model Selection**
        → **7. Loan Prediction**
        """
    )

    st.subheader("📌 Project Features")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        - Loan Approval Prediction
        - Multiple ML Model Comparison
        - Automatic Best Model Selection
        - Risk Level Classification
        """)

    with col2:
        st.write("""
        - Applicant Summary
        - Exploratory Data Analysis
        - Accuracy, Precision, Recall and F1 Score
        - Interactive Streamlit Dashboard
        """)


# ==========================================
# LOAN PREDICTION PAGE
# ==========================================

elif page == "🔮 Loan Prediction":

    st.title("🔮 Loan Approval Prediction")

    st.write(
        "Enter the applicant details below to predict the loan status."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)


    # ------------------------------------------
    # COLUMN 1
    # ------------------------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        married = st.selectbox(
            "Married",
            ["no", "yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["graduate", "not graduate"]
        )


    # ------------------------------------------
    # COLUMN 2
    # ------------------------------------------

    with col2:

        self_employed = st.selectbox(
            "Self Employed",
            ["no", "yes"]
        )

        applicantincome = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        coapplicantincome = st.number_input(
            "Co-applicant Income",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

        loanamount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=120.0,
            step=10.0
        )


    # ------------------------------------------
    # COLUMN 3
    # ------------------------------------------

    with col3:

        loan_amount_term = st.selectbox(
            "Loan Amount Term",
            [
                12.0,
                36.0,
                60.0,
                84.0,
                120.0,
                180.0,
                240.0,
                300.0,
                360.0,
                480.0
            ],
            index=8
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x:
                "Good (1)" if x == 1.0 else "Poor (0)"
        )

        property_area = st.selectbox(
            "Property Area",
            ["rural", "semiurban", "urban"],
            index=2
        )


    st.divider()


    # ==========================================
    # PREDICTION
    # ==========================================

    if st.button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "gender": gender,
            "married": married,
            "dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "applicantincome": applicantincome,
            "coapplicantincome": coapplicantincome,
            "loanamount": loanamount,
            "loan_amount_term": loan_amount_term,
            "credit_history": credit_history,
            "property_area": property_area
        }])


        # Encode categorical columns
        for column, encoder in label_encoders.items():

            input_data[column] = encoder.transform(
                input_data[column]
            )


        # Put features in correct order
        input_data = input_data[feature_names]


        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]


        st.divider()

        st.subheader("📋 Prediction Result")


        # ------------------------------------------
        # APPROVED
        # ------------------------------------------

        if prediction == 1:

            approval_probability = probability[1] * 100

            st.success("✅ LOAN APPROVED")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Approval Probability",
                    f"{approval_probability:.2f}%"
                )

            with col2:

                if approval_probability >= 75:
                    st.success("🟢 Low Risk")

                elif approval_probability >= 50:
                    st.warning("🟡 Medium Risk")

                else:
                    st.error("🔴 High Risk")


        # ------------------------------------------
        # REJECTED
        # ------------------------------------------

        else:

            rejection_probability = probability[0] * 100

            st.error("❌ LOAN REJECTED")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Rejection Probability",
                    f"{rejection_probability:.2f}%"
                )

            with col2:
                st.error("🔴 High Risk")


        # ==========================================
        # APPLICANT SUMMARY
        # ==========================================

        st.divider()

        st.subheader("👤 Applicant Summary")

        summary_data = {
            "Gender": gender.title(),
            "Married": married.title(),
            "Dependents": dependents,
            "Education": education.title(),
            "Self Employed": self_employed.title(),
            "Applicant Income": f"₹{applicantincome:,.0f}",
            "Co-applicant Income": f"₹{coapplicantincome:,.0f}",
            "Loan Amount": f"₹{loanamount:,.0f}",
            "Loan Amount Term": f"{loan_amount_term:.0f} Months",
            "Credit History": "Good" if credit_history == 1 else "Poor",
            "Property Area": property_area.title()
        }

        summary_df = pd.DataFrame(
            list(summary_data.items()),
            columns=["Field", "Value"]
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# MODEL COMPARISON PAGE
# ==========================================

elif page == "📊 Model Comparison":

    st.title("📊 Machine Learning Model Comparison")

    st.write(
        "Five classification algorithms were trained and evaluated."
    )

    st.divider()

    st.subheader("📋 Model Performance")

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )


    # Accuracy chart
    st.subheader("📊 Accuracy Comparison")

    chart_data = model_results.set_index("Model")

    st.bar_chart(
        chart_data["Accuracy"]
    )


    # Best Model
    best_model = model_results.iloc[0]

    st.divider()

    st.success(
        f"🏆 Best Model: {best_model['Model']} | "
        f"Accuracy: {best_model['Accuracy']}%"
    )


    st.subheader("📖 Evaluation Metrics")

    st.write("""
    **Accuracy:** Percentage of total predictions that are correct.

    **Precision:** Accuracy of positive loan approval predictions.

    **Recall:** Ability of the model to identify actual approved loans.

    **F1 Score:** Harmonic balance between Precision and Recall.
    """)


# ==========================================
# EDA DASHBOARD PAGE
# ==========================================

elif page == "📈 EDA Dashboard":

    st.title("📈 Exploratory Data Analysis Dashboard")

    st.write(
        "Visual analysis of the Loan Approval Prediction dataset."
    )

    st.divider()


    # Create a copy for visualization
    eda_df = df.copy()

    eda_df["loan_status"] = (
        eda_df["loan_status"]
        .astype(str)
        .str.upper()
    )


    # ==========================================
    # 1. LOAN STATUS DISTRIBUTION
    # ==========================================

    st.subheader("1️⃣ Loan Approval Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        data=eda_df,
        x="loan_status",
        ax=ax
    )

    ax.set_xlabel("Loan Status (Y = Approved, N = Rejected)")
    ax.set_ylabel("Number of Applicants")
    ax.set_title("Loan Approval Distribution")

    st.pyplot(fig)

    plt.close(fig)


    # ==========================================
    # 2. APPLICANT INCOME DISTRIBUTION
    # ==========================================

    st.subheader("2️⃣ Applicant Income Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        data=eda_df,
        x="applicantincome",
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Number of Applicants")
    ax.set_title("Applicant Income Distribution")

    st.pyplot(fig)

    plt.close(fig)


    # ==========================================
    # 3. LOAN AMOUNT DISTRIBUTION
    # ==========================================

    st.subheader("3️⃣ Loan Amount Distribution")

    fig, ax = plt.subplots()

    sns.histplot(
        data=eda_df,
        x="loanamount",
        kde=True,
        ax=ax
    )

    ax.set_xlabel("Loan Amount")
    ax.set_ylabel("Number of Applicants")
    ax.set_title("Loan Amount Distribution")

    st.pyplot(fig)

    plt.close(fig)


    # ==========================================
    # 4. CREDIT HISTORY VS LOAN STATUS
    # ==========================================

    st.subheader("4️⃣ Credit History vs Loan Approval")

    fig, ax = plt.subplots()

    sns.countplot(
        data=eda_df,
        x="credit_history",
        hue="loan_status",
        ax=ax
    )

    ax.set_xlabel("Credit History")
    ax.set_ylabel("Number of Applicants")
    ax.set_title("Credit History vs Loan Approval")

    st.pyplot(fig)

    plt.close(fig)


    # ==========================================
    # 5. PROPERTY AREA VS LOAN STATUS
    # ==========================================

    st.subheader("5️⃣ Property Area vs Loan Approval")

    fig, ax = plt.subplots()

    sns.countplot(
        data=eda_df,
        x="property_area",
        hue="loan_status",
        ax=ax
    )

    ax.set_xlabel("Property Area")
    ax.set_ylabel("Number of Applicants")
    ax.set_title("Property Area vs Loan Approval")

    st.pyplot(fig)

    plt.close(fig)
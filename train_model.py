import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("loan_approval_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE LOAN ID
# ==========================================

df = df.drop("loan_id", axis=1)


# ==========================================
# 3. CONVERT TARGET COLUMN
# ==========================================

df["loan_status"] = df["loan_status"].str.lower().map({
    "y": 1,
    "n": 0
})


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

for column in df.columns:

    # Check whether the column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())

    # Otherwise it is categorical/string
    else:
        df[column] = df[column].fillna(df[column].mode()[0])


print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("loan_status", axis=1)
y = df["loan_status"]


# ==========================================
# 6. ENCODE CATEGORICAL COLUMNS
# ==========================================

label_encoders = {}

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns


for column in categorical_columns:

    le = LabelEncoder()

    X[column] = le.fit_transform(X[column])

    label_encoders[column] = le


print("\nProcessed Data:")
print(X.head())


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. DEFINE MACHINE LEARNING MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])
}


# ==========================================
# 9. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    trained_models[name] = model


# ==========================================
# 10. CREATE MODEL COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)


print("\n========================================")
print("         MODEL COMPARISON")
print("========================================")

print(results_df.to_string(index=False))


# ==========================================
# 11. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# ==========================================
# 12. CREATE MODEL FOLDER
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# 13. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "model/best_model.pkl"
)


# ==========================================
# 14. SAVE LABEL ENCODERS
# ==========================================

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)


# ==========================================
# 15. SAVE FEATURE NAMES
# ==========================================

joblib.dump(
    list(X.columns),
    "model/feature_names.pkl"
)


# ==========================================
# 16. SAVE MODEL RESULTS
# ==========================================

results_df.to_csv(
    "model/model_results.csv",
    index=False
)


# ==========================================
# 17. SUCCESS MESSAGE
# ==========================================

print("\n========================================")
print("PROJECT MODEL TRAINING COMPLETED!")
print("========================================")

print("\nBest Model Saved:")
print("model/best_model.pkl")

print("\nSupporting Files Saved:")
print("model/label_encoders.pkl")
print("model/feature_names.pkl")
print("model/model_results.csv")
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# File paths
data_path = r"C:\Users\zohaib khan\OneDrive\Desktop\USE ME\dump\zk\smoke_detection.csv"
model_file = "fire_model.pkl"
scaler_file = "fire_scaler.pkl"
encoder_file = "fire_label_encoders.pkl"

# Streamlit page config
st.set_page_config(page_title="🔥 Smoke Detection", layout="wide")
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["📊 Model Evaluation", "🔮 Predict on New Data"])

# Load dataset
df = pd.read_csv(data_path, encoding='latin1')
target_col = "Fire Alarm"
X = df.drop(target_col, axis=1)
y = df[target_col]

# Encode categorical columns
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
joblib.dump(label_encoders, encoder_file)

# Encode target if it's not already numeric
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    joblib.dump(target_encoder, "target_encoder.pkl")
    class_names = target_encoder.classes_
else:
    class_names = ["0", "1"]  # 0 = No Fire, 1 = Fire

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, scaler_file)

# Load or train model
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, model_file)

# Page 1: Model Evaluation
if page == "📊 Model Evaluation":
    st.title("📊 Smoke Detection Model Evaluation")

    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(df.head())

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", round(accuracy_score(y_train, y_pred_train), 3))
    col2.metric("Testing Accuracy", round(accuracy_score(y_test, y_pred_test), 3))

    st.markdown("### 📋 Classification Report")
    st.text(classification_report(y_test, y_pred_test, target_names=["No Fire (0)", "Fire (1)"]))

    st.markdown("### 🔲 Confusion Matrix")
    fig_cm, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Fire (0)", "Fire (1)"], yticklabels=["No Fire (0)", "Fire (1)"])
    st.pyplot(fig_cm)

# Page 2: Prediction
if page == "🔮 Predict on New Data":
    st.title("🔮 Predict Fire Alarm (0 = No Fire, 1 = Fire)")

    input_data = {}
    for col in X.columns:
        if col in categorical_cols:
            options = label_encoders[col].classes_.tolist()
            input_data[col] = st.selectbox(col, options)
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Predict Fire Alarm"):
        input_df = pd.DataFrame([input_data])
        label_encoders = joblib.load(encoder_file)
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])

        scaler = joblib.load(scaler_file)
        input_scaled = scaler.transform(input_df)
        model = joblib.load(model_file)
        pred = model.predict(input_scaled)[0]

        st.subheader("📢 Prediction Result")
        if pred == 1:
            st.error("🔥 Fire Detected! (1)")
        else:
            st.success("✅ No Fire Detected (0)")


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# Page config
# ============================
st.set_page_config(page_title="Medical Risk Predictor", layout="wide")
st.title("🩺 Medical Risk Score Predictor")
st.write("Predict **Heart Disease**, **Diabetes**, and **Stroke** risk with explainable ML.")

# ============================
# Load models, encoders, features
# ============================
@st.cache_resource
def load_assets():
    assets = {}
    assets["heart_model"] = joblib.load("models/heart_disease_xgb.joblib")
    assets["heart_enc"] = joblib.load("models/heart_encoder.joblib")
    assets["heart_feats"] = joblib.load("models/heart_features.joblib")

    assets["diab_model"] = joblib.load("models/diabetes_xgb.joblib")
    assets["diab_enc"] = joblib.load("models/diabetes_encoder.joblib")
    assets["diab_feats"] = joblib.load("models/diabetes_features.joblib")

    assets["stroke_model"] = joblib.load("models/stroke_xgb.joblib")
    assets["stroke_enc"] = joblib.load("models/stroke_encoder.joblib")
    assets["stroke_feats"] = joblib.load("models/stroke_features.joblib")
    return assets

A = load_assets()

# ============================
# Sidebar inputs
# ============================
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 90, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
BMI = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
systolic_bp = st.sidebar.slider("Systolic BP", 90, 200, 120)
diastolic_bp = st.sidebar.slider("Diastolic BP", 60, 120, 80)
cholesterol = st.sidebar.slider("Cholesterol", 100, 400, 200)
glucose = st.sidebar.slider("Glucose", 60, 300, 100)
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
physical_activity = st.sidebar.selectbox("Physically Active", ["No", "Yes"])

# ============================
# Feature engineering
# ============================
gender_val = 1 if gender == "Male" else 0
smoking_val = 1 if smoking == "Yes" else 0
pa_val = 1 if physical_activity == "Yes" else 0

BMI_cat = (
    "Underweight" if BMI < 18.5 else
    "Normal" if BMI < 25 else
    "Overweight" if BMI < 30 else
    "Obese"
)

pulse_pressure = systolic_bp - diastolic_bp
age_decade = int(age // 10)

base_input = pd.DataFrame([{
    "age": age,
    "age_decade": age_decade,
    "gender": gender_val,
    "BMI": BMI,
    "BMI_cat": BMI_cat,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "pulse_pressure": pulse_pressure,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "smoking": smoking_val,
    "physical_activity": pa_val
}])

# Missing flags = 0
for c in list(base_input.columns):
    base_input[f"{c}_missing"] = 0

# ============================
# Helper functions
# ============================
def encode_and_align(df, encoder, feature_list):
    cat_cols = [c for c in ["gender", "BMI_cat"] if c in df.columns]
    df_enc = df.copy()
    df_enc[cat_cols] = encoder.transform(df_enc[cat_cols])

    aligned = pd.DataFrame(0, index=[0], columns=feature_list)
    for c in df_enc.columns:
        if c in aligned.columns:
            aligned[c] = df_enc[c].values
    return aligned

def overall_health_score(h, d, s):
    return round((0.4*h + 0.35*d + 0.25*s) * 100, 2)

def local_feature_impact(model, base_df, encoder, feature_list, feature_changes):
    base_X = encode_and_align(base_df, encoder, feature_list)
    base_prob = model.predict_proba(base_X)[:, 1][0]

    impacts = []
    for feature, new_value in feature_changes.items():
        temp_df = base_df.copy()
        temp_df[feature] = new_value
        X_temp = encode_and_align(temp_df, encoder, feature_list)
        new_prob = model.predict_proba(X_temp)[:, 1][0]
        impacts.append((feature, new_prob - base_prob))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return impacts

# ============================
# Predictions
# ============================
Xh = encode_and_align(base_input, A["heart_enc"], A["heart_feats"])
Xd = encode_and_align(base_input, A["diab_enc"], A["diab_feats"])
Xs = encode_and_align(base_input, A["stroke_enc"], A["stroke_feats"])

heart_prob = A["heart_model"].predict_proba(Xh)[:,1][0]
diab_prob  = A["diab_model"].predict_proba(Xd)[:,1][0]
stroke_prob= A["stroke_model"].predict_proba(Xs)[:,1][0]

overall = overall_health_score(heart_prob, diab_prob, stroke_prob)

# ============================
# Results
# ============================
st.subheader("🔍 Risk Predictions")
c1, c2, c3, c4 = st.columns(4)
c1.metric("❤️ Heart Disease", f"{heart_prob*100:.1f}%")
c2.metric("🔥 Diabetes", f"{diab_prob*100:.1f}%")
c3.metric("🧠 Stroke", f"{stroke_prob*100:.1f}%")
c4.metric("📊 Overall Health Risk", f"{overall}%")

st.caption("⚠️ This is a risk score for screening/education only. Not a medical diagnosis.")

# ============================
# Simple Explainability (NO SHAP)
# ============================
st.subheader("🧠 Explainability — Heart Disease (Simple & Stable)")

feature_tests = {
    "age": age + 5,
    "BMI": BMI + 2,
    "systolic_bp": systolic_bp + 10,
    "cholesterol": cholesterol + 20,
    "glucose": glucose + 20,
    "smoking": 1 - smoking_val,
    "physical_activity": 1 - pa_val
}

impacts = local_feature_impact(
    A["heart_model"],
    base_input,
    A["heart_enc"],
    A["heart_feats"],
    feature_tests
)

with st.expander("Show explanation"):
    st.write("Top factors affecting this prediction:")
    for feat, delta in impacts[:5]:
        direction = "increased" if delta > 0 else "reduced"
        st.write(f"- **{feat}** {direction} risk by {abs(delta)*100:.1f}%")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ── Load Model ───────────────────────────────────────────────────
model  = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# ── Title ────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; color:#e74c3c;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Enter patient details below to predict heart disease risk</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar Inputs ───────────────────────────────────────────────
st.sidebar.header("🧾 Patient Details")

age      = st.sidebar.slider("Age", 20, 80, 50)
sex      = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp       = st.sidebar.selectbox("Chest Pain Type", [
               "0 - No Pain",
               "1 - Mild Pain",
               "2 - Moderate Pain",
               "3 - Severe Pain"])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol     = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg  = st.sidebar.selectbox("Resting ECG", [
               "0 - Normal",
               "1 - ST-T Abnormality",
               "2 - Left Ventricular Hypertrophy"])
thalach  = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang    = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak  = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
slope    = st.sidebar.selectbox("Slope of ST Segment", [
               "0 - Upsloping",
               "1 - Flat",
               "2 - Downsloping"])
ca       = st.sidebar.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
thal     = st.sidebar.selectbox("Thalassemia", [
               "0 - Normal",
               "1 - Fixed Defect",
               "2 - Reversible Defect",
               "3 - Unknown"])

# ── Convert Inputs ───────────────────────────────────────────────
sex_val     = 1 if sex == "Male" else 0
cp_val      = int(cp[0])
fbs_val     = 1 if fbs == "Yes" else 0
restecg_val = int(restecg[0])
exang_val   = 1 if exang == "Yes" else 0
slope_val   = int(slope[0])
thal_val    = int(thal[0])

input_data = pd.DataFrame([[
    age, sex_val, cp_val, trestbps, chol, fbs_val,
    restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val
]], columns=['age','sex','cp','trestbps','chol','fbs',
             'restecg','thalach','exang','oldpeak','slope','ca','thal'])

input_scaled = scaler.transform(input_data)

# ── Predict Button ───────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict Now", use_container_width=True)

st.markdown("---")

if predict_btn:
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("## ❤️‍🔥 Heart Disease Detected")
            st.error(f"### Confidence: {probability*100:.1f}%")
            st.warning("⚠️ Please consult a cardiologist immediately.")
        else:
            st.success("## ✅ No Heart Disease Detected")
            st.success(f"### Confidence: {(1-probability)*100:.1f}%")
            st.info("💚 Keep maintaining a healthy lifestyle!")

    with col2:
        # Confidence Gauge
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ['#2ecc71', '#e74c3c']
        values = [1 - probability, probability]
        labels = ['No Disease', 'Disease']
        ax.pie(values, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Risk Distribution')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Patient Summary ──────────────────────────────────────────
    st.subheader("📋 Patient Summary")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Age", f"{age} yrs")
    summary_col2.metric("Cholesterol", f"{chol} mg/dl")
    summary_col3.metric("Blood Pressure", f"{trestbps} mm Hg")
    summary_col4.metric("Max Heart Rate", f"{thalach} bpm")

    st.markdown("---")

    # ── Load dataset for charts ──────────────────────────────────
    try:
        df = pd.read_csv('heart.csv')
        X  = df.drop('target', axis=1)
        y  = df['target']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'])
            ax2.set_ylabel('Actual')
            ax2.set_xlabel('Predicted')
            ax2.set_title('Confusion Matrix')
            st.pyplot(fig2)
            plt.close()

        with chart_col2:
            st.subheader("📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc     = auc(fpr, tpr)
            fig3, ax3   = plt.subplots(figsize=(5, 4))
            ax3.plot(fpr, tpr, color='#e74c3c', lw=2,
                     label=f'ROC Curve (AUC = {roc_auc:.2f})')
            ax3.plot([0,1],[0,1], color='gray', linestyle='--')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve — Random Forest')
            ax3.legend()
            st.pyplot(fig3)
            plt.close()

    except:
        pass

else:
    # ── Default screen before prediction ────────────────────────
    st.markdown("### 👈 Fill in the patient details on the left sidebar and click Predict!")
    
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1**\n\nFill patient details in the sidebar")
    col2.info("**Step 2**\n\nClick the Predict button")
    col3.info("**Step 3**\n\nSee result + charts instantly")
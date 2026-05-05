import streamlit as st
import joblib

# Load model
model = joblib.load("../model/model.pkl")

st.title("💳 Credit Card Fraud Detection")

#Initialize session state
if "features" not in st.session_state:
    st.session_state.features = [0.0] * 30

#Sample Data
normal_sample = [10000] + [-1.2,0.5,1.8,0.3,-0.4,0.2,-0.1,0.05,0.3,-0.2,
                          0.1,0.2,-0.1,0.05,0.3,-0.2,0.1,-0.1,0.2,0.05,
                          -0.1,0.2,-0.05,0.1,0.2,-0.1,0.01,0.02] + [120]

fraud_sample = [50000] + [-5.0,4.5,-6.0,5.5,-3.5,2.5,-4.0,3.0,-5.0,4.5,
                         -6.0,5.5,-3.0,-7.0,2.0,-4.0,-6.5,3.5,-2.5,2.0,
                         -3.0,2.5,-2.0,1.5,-1.5,1.0,-0.5,0.5] + [2500]

#Buttons
col1, col2 = st.columns(2)

if col1.button("🟢 Load Normal Data"):
    st.session_state.features = normal_sample

if col2.button("🔴 Load Fraud Data"):
    st.session_state.features = fraud_sample

#Input Fields
features = []
labels = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

for i, label in enumerate(labels):
    val = st.number_input(label, value=float(st.session_state.features[i]))
    features.append(val)

#Predict
if st.button("Predict"):
    prediction = model.predict([features])

    if prediction[0] == 1:
        st.error("⚠️ Fraud Transaction")
    else:
        st.success("✅ Normal Transaction")
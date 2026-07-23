import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Quality Escape Prevention",
    page_icon="🏭",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

model = joblib.load(MODEL_PATH)

# ==========================================================
# TITLE
# ==========================================================

st.title("🏭 Quality Escape Prevention")

st.markdown(
"""
### Manufacturing Defect Prediction System

This application predicts whether a manufactured product will:

- ✅ PASS Quality Inspection
- ❌ FAIL Quality Inspection

Enter the sensor values below and click **Predict**.
"""
)

st.info(
    "Demo Version: This application uses 5 selected sensor features from the SECOM Manufacturing Dataset."
)

# ==========================================================
# INPUT SECTION
# ==========================================================
st.subheader("Enter Manufacturing Sensor Values")

col1, col2 = st.columns(2)

with col1:
    f0 = st.number_input("🌡️ Temperature", value=0.0)
    f1 = st.number_input("⚙️ Pressure", value=0.0)
    f2 = st.number_input("💨 Humidity", value=0.0)
    f3 = st.number_input("⚡ Voltage", value=0.0)
    f4 = st.number_input("🔌 Current", value=0.0)


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button("Predict"):

    input_data = pd.DataFrame({
        "0": [f0],
        "1": [f1],
        "2": [f2],
        "3": [f3],
        "4": [f4],
     
    })

    prediction = model.predict(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❌ FAIL")
        st.write("The product is predicted to **FAIL** quality inspection.")
    else:
        st.success("✅ PASS")
        st.write("The product is predicted to **PASS** quality inspection.")

    st.subheader("Input Summary")
    st.dataframe(input_data)
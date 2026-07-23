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
MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "notebooks", "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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

**📊 Expected Ranges (from training data):**
- **PASS**: Temperature 2900-3100, Pressure 2400-2600, Humidity 2100-2300, Voltage 1200-1500, Current 1.0-1.6
- **FAIL**: Temperature 2400-2700, Pressure 1800-2200, Humidity 1600-2000, Voltage 800-1200, Current 0.3-0.8
"""
)

st.info(
    "Demo Version: This application uses 5 selected sensor features from the SECOM Manufacturing Dataset."
)

# Add sidebar with ranges
with st.sidebar:
    st.subheader("📋 Reference Ranges")
    
    st.write("**✅ PASS Ranges:**")
    st.write("""
    - Temperature: 2900-3100
    - Pressure: 2400-2600  
    - Humidity: 2100-2300
    - Voltage: 1200-1500
    - Current: 1.0-1.6
    """)
    
    st.write("**❌ FAIL Ranges:**")
    st.write("""
    - Temperature: 2400-2700
    - Pressure: 1800-2200
    - Humidity: 1600-2000
    - Voltage: 800-1200
    - Current: 0.3-0.8
    """)

# ==========================================================
# INPUT SECTION
# ==========================================================
st.subheader("Enter Manufacturing Sensor Values")

# Example buttons
col_example1, col_example2 = st.columns(2)

with col_example1:
    if st.button("📊 Example: PASS"):
        st.session_state.pass_example = True
        
with col_example2:
    if st.button("⚠️ Example: FAIL"):
        st.session_state.fail_example = True

# Set example values
if "pass_example" in st.session_state and st.session_state.pass_example:
    f0 = st.session_state.get("f0", 3000.0)
    f1 = st.session_state.get("f1", 2500.0)
    f2 = st.session_state.get("f2", 2200.0)
    f3 = st.session_state.get("f3", 1400.0)
    f4 = st.session_state.get("f4", 1.0)
    st.session_state.pass_example = False
elif "fail_example" in st.session_state and st.session_state.fail_example:
    f0 = 2550.0
    f1 = 2000.0
    f2 = 1800.0
    f3 = 1000.0
    f4 = 0.5
    st.session_state.fail_example = False
else:
    f0 = 0.0
    f1 = 0.0
    f2 = 0.0
    f3 = 0.0
    f4 = 0.0

col1, col2 = st.columns(2)

with col1:
    f0 = st.number_input("🌡️ Temperature", value=f0)
    f1 = st.number_input("⚙️ Pressure", value=f1)
    f2 = st.number_input("💨 Humidity", value=f2)
    f3 = st.number_input("⚡ Voltage", value=f3)
    f4 = st.number_input("🔌 Current", value=f4)


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

    # Scale the input data using the saved scaler
    input_scaled = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

    prediction = model.predict(input_scaled_df)[0]

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
    
    st.subheader("Scaled Input (Used for Prediction)")
    st.dataframe(input_scaled_df)
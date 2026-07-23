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
- **FAIL**: Temperature 2848-3267, Pressure 2286-2714, Humidity 2139-2315, Voltage 900-2360, Current 0.68-2.42
- **PASS**: Temperature 2743-3356, Pressure 2159-2846, Humidity 2061-2315, Voltage 0-3715, Current 0.68-114.5

*Note: The ranges overlap significantly. The key differentiator is Current - FAIL tends to have lower current values.*
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
    - Temperature: 2743-3356
    - Pressure: 2159-2846  
    - Humidity: 2061-2315
    - Voltage: 0-3715
    - Current: 0.68-114.54
    """)
    
    st.write("**❌ FAIL Ranges:**")
    st.write("""
    - Temperature: 2848-3267
    - Pressure: 2286-2714
    - Humidity: 2139-2315
    - Voltage: 900-2360
    - Current: 0.68-2.42
    """)
    
    st.write("**💡 Key Insight:**")
    st.write("Current is the main differentiator. FAIL has lower current (0.68-2.42) vs PASS (0.68-114.54)")

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
    f0 = st.session_state.get("f0", 3015.0)
    f1 = st.session_state.get("f1", 2496.0)
    f2 = st.session_state.get("f2", 2201.0)
    f3 = st.session_state.get("f3", 1398.0)
    f4 = st.session_state.get("f4", 4.38)
    st.session_state.pass_example = False
elif "fail_example" in st.session_state and st.session_state.fail_example:
    f0 = 3008.0
    f1 = 2495.0
    f2 = 2200.0
    f3 = 1356.0
    f4 = 1.3
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
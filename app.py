import streamlit as st
import pandas as pd
import joblib

# Load the permanently saved model and scaler
model = joblib.load('exoplanet_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the feature columns in the exact order your model expects
feature_cols = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
    'koi_fpflag_ec', 'koi_period', 'koi_period_err1', 'koi_period_err2',
    'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact',
    'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1',
    'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
    'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq', 'koi_insol',
    'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_steff',
    'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1',
    'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra',
    'dec', 'koi_kepmag'
]

# --- Streamlit User Interface ---
st.title("Exoplanet Prediction App 🔭")
st.write("Enter the astronomical data below to predict if an object is a confirmed exoplanet or a false positive.")

# Create a sidebar for user inputs
st.sidebar.header("Input Features")
input_data = {}

# Use a loop to create number inputs for all features in the sidebar
for feature in feature_cols:
    input_data[feature] = st.sidebar.number_input(f'{feature}', value=0.0, format="%.4f")

# Create a "Predict" button
if st.button("Predict"):
    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_cols] # Ensure column order

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction and get probabilities
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Display the results
    st.subheader("Prediction Result")
    result_class = "Confirmed Exoplanet" if prediction[0] == 1 else "False Positive"
    confidence = prediction_proba[0][prediction[0]] * 100

    if result_class == "Confirmed Exoplanet":
        st.success(f"The model predicts this is a **{result_class}** with **{confidence:.2f}%** confidence.")
    else:
        st.error(f"The model predicts this is a **{result_class}** with **{confidence:.2f}%** confidence.")
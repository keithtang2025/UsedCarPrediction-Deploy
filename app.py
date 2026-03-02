# app.py
from pathlib import Path
import joblib
import requests
import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("Used Car Price Predictor (Australia)")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

# If you prefer Streamlit Secrets, set MODEL_URL in Streamlit Cloud -> Settings -> Secrets
# Otherwise this default URL will be used.
DEFAULT_MODEL_URL = "https://github.com/keithtang2025/UsedCarPrediction-Deploy/releases/download/v1/model.pkl"


@st.cache_resource
def load_model():
    # 1) Get URL from secrets if provided, otherwise use default
    model_url = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

    # 2) Download model if missing
    if not MODEL_PATH.exists():
        st.info("Downloading model (first run only)...")
        with st.spinner("Downloading..."):
            r = requests.get(model_url, stream=True, timeout=300)
            r.raise_for_status()

            # Write in chunks to avoid memory issues
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

        st.success("Model downloaded!")

    # 3) Load model
    return joblib.load(MODEL_PATH)


# Load model pipeline
pipe = load_model()

st.write("Enter a car’s details and click **Predict**.")

# ----------------------------
# Input UI (adjust to your real feature columns)
# ----------------------------
brand = st.text_input("Brand", "Toyota")
year = st.number_input("Year", min_value=1980, max_value=2026, value=2018)
odometer = st.number_input("Odometer (km)", min_value=0, value=60000)
transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "Hybrid", "Electric"])

if st.button("Predict"):
    # IMPORTANT:
    # These column names MUST match what your model was trained on.
    # If your CSV uses different names, change the keys below accordingly.
    X = pd.DataFrame([{
        "Brand": brand,
        "Year": year,
        "Odometer": odometer,
        "Transmission": transmission,
        "FuelType": fuel
    }])

    try:
        pred = pipe.predict(X)[0]
        st.success(f"Estimated price: ${pred:,.0f}")
    except Exception as e:
        st.error("Prediction failed — input columns likely don’t match the model’s expected columns.")
        st.code(str(e))

        st.write("Tip: Print the model’s expected feature names (if available) and align your UI inputs.")

# app.py
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("Used Car Price Predictor (Australia)")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipe = load_model()

st.write("Enter a car’s details and get a predicted price.")

# IMPORTANT: replace these example fields with the real columns your model expects
brand = st.text_input("Brand", "Toyota")
year = st.number_input("Year", min_value=1980, max_value=2026, value=2018)
km = st.number_input("Odometer (km)", min_value=0, value=60000)
transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "Hybrid", "Electric"])

if st.button("Predict"):
    X = pd.DataFrame([{
        "Brand": brand,
        "Year": year,
        "Odometer": km,
        "Transmission": transmission,
        "FuelType": fuel,
    }])

    try:
        pred = pipe.predict(X)[0]
        st.success(f"Estimated price: ${pred:,.0f}")
    except Exception as e:
        st.error("Your input columns don’t match the model’s expected columns.")
        st.code(str(e))
        st.write("Fix: make `app.py` build a DataFrame with exactly the same columns used in training.")

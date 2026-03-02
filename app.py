# app.py
from pathlib import Path
import joblib
import requests
import pandas as pd
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("Used Car Price Predictor (Australia)")

# ----------------------------
# Paths / Model URL
# ----------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

# You can override this in Streamlit Cloud -> Settings -> Secrets:
# MODEL_URL = "https://github.com/keithtang2025/UsedCarPrediction-Deploy/releases/download/v1/model.pkl"
DEFAULT_MODEL_URL = "https://github.com/keithtang2025/UsedCarPrediction-Deploy/releases/download/v1/model.pkl"


@st.cache_resource
def load_model():
    """
    Download the model from GitHub Releases on first run, then load it.
    Cached so the model doesn't reload every rerun.
    """
    model_url = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

    if not MODEL_PATH.exists():
        st.info("Downloading model (first run only)...")
        with st.spinner("Downloading model..."):
            r = requests.get(model_url, stream=True, timeout=300)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
        st.success("Model downloaded!")

    return joblib.load(MODEL_PATH)


# Load pipeline
pipe = load_model()

st.write("Fill in the vehicle details and click **Predict**.")

# ----------------------------
# Input UI (matches your model’s required columns)
# ----------------------------
st.subheader("Vehicle details")

UsedOrNew = st.selectbox("UsedOrNew", ["Used", "New"])

DriveType = st.selectbox(
    "DriveType",
    ["FWD", "RWD", "AWD", "4WD", "Other"]
)

ColourExtInt = st.text_input(
    "ColourExtInt (e.g., 'White / Black')",
    "White / Black"
)

BodyType = st.text_input(
    "BodyType (e.g., 'Sedan', 'Hatch', 'SUV')",
    "Sedan"
)

Engine = st.text_input(
    "Engine (e.g., '2.0L', '3.0L Turbo')",
    "2.0L"
)

Doors = st.number_input(
    "Doors",
    min_value=2, max_value=6, value=4, step=1
)

CylindersinEngine = st.number_input(
    "CylindersinEngine",
    min_value=0, max_value=16, value=4, step=1
)

Car_Suv = st.selectbox(
    "Car/Suv",
    ["Car", "Suv"]
)

FuelConsumption = st.number_input(
    "FuelConsumption (L/100km)",
    min_value=0.0, max_value=50.0, value=7.5, step=0.1
)

Kilometres = st.number_input(
    "Kilometres",
    min_value=0, value=60000, step=1000
)

Location = st.text_input(
    "Location (e.g., 'NSW - Sydney')",
    "NSW - Sydney"
)

Seats = st.number_input(
    "Seats",
    min_value=1, max_value=12, value=5, step=1
)

Model = st.text_input(
    "Model (e.g., 'Corolla')",
    "Corolla"
)

Title = st.text_input(
    "Title (listing title / description)",
    "2018 Toyota Corolla Auto Hatch"
)

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict"):
    # IMPORTANT: dict keys must match training columns EXACTLY, including "Car/Suv"
    X = pd.DataFrame([{
        "UsedOrNew": UsedOrNew,
        "DriveType": DriveType,
        "ColourExtInt": ColourExtInt,
        "BodyType": BodyType,
        "Engine": Engine,
        "Doors": Doors,
        "CylindersinEngine": CylindersinEngine,
        "Car/Suv": Car_Suv,
        "FuelConsumption": FuelConsumption,
        "Kilometres": Kilometres,
        "Location": Location,
        "Seats": Seats,
        "Model": Model,
        "Title": Title
    }])

    try:
        pred = pipe.predict(X)[0]
        st.success(f"Estimated price: ${pred:,.0f}")
    except Exception as e:
        st.error("Prediction failed. Input format may not match the model expectation.")
        st.code(str(e))
        st.write("Debug — columns sent:", list(X.columns))
        st.write(X.head(1))

from pathlib import Path
import re
import joblib
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("Used Car Price Predictor (Australia)")

ROOT = Path(__file__).resolve().parent

# ----------------------------
# Model download + load
# ----------------------------
MODEL_PATH = ROOT / "model.pkl"
DEFAULT_MODEL_URL = "https://github.com/keithtang2025/UsedCarPrediction-Deploy/releases/download/v1/model.pkl"

@st.cache_resource
def load_model():
    model_url = st.secrets.get("MODEL_URL", DEFAULT_MODEL_URL)

    if not MODEL_PATH.exists():
        st.info("Downloading model (first run only)...")
        with st.spinner("Downloading model..."):
            r = requests.get(model_url, stream=True, timeout=300)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        st.success("Model downloaded!")

    return joblib.load(MODEL_PATH)

pipe = load_model()

# ----------------------------
# Load CSV for validation + dropdown options
# ----------------------------
CSV_PATH = ROOT / "AustralianVehiclePrices.csv"

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

df = load_csv(CSV_PATH)

EXPECTED_FEATURES = [
    "Brand","FuelType","Year","Transmission",
    "UsedOrNew","DriveType","ColourExtInt","BodyType","Engine",
    "Doors","CylindersinEngine","Car/Suv","FuelConsumption",
    "Kilometres","Location","Seats","Model","Title"
]

def safe_unique(col: str, max_n: int | None = None):
    if df.empty or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).unique().tolist()
    vals = sorted(vals)
    if max_n is not None:
        return vals[:max_n]
    return vals

def parse_first_number(s: str) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if s in ["", "-", "- / -"]:
        return None
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

# Dataset-driven ranges (for validation)
if not df.empty:
    year_min = int(np.nanmin(df["Year"].values))
    year_max = int(np.nanmax(df["Year"].values))

    km_series = df["Kilometres"].apply(parse_first_number)
    km_p99 = float(np.nanpercentile(km_series.dropna(), 99)) if km_series.notna().any() else 400_000.0

    fc_series = df["FuelConsumption"].apply(parse_first_number)
    fc_p99 = float(np.nanpercentile(fc_series.dropna(), 99)) if fc_series.notna().any() else 15.0
else:
    year_min, year_max = 1990, 2023
    km_p99, fc_p99 = 400_000.0, 15.0

# ----------------------------
# UI
# ----------------------------
st.write("Fill in the vehicle details and click **Predict**.")

with st.expander("Data/model notes (optional)", expanded=False):
    st.write("- This UI validates inputs using your CSV ranges and uses dropdowns where possible.")
    st.write("- If you enter categories not seen in training, the model will still run (unknown categories are ignored).")

st.subheader("Vehicle details")

# Dropdowns from CSV (searchable in Streamlit)
brand_options = safe_unique("Brand")
Brand = st.selectbox("Brand", brand_options if brand_options else ["Toyota", "Honda", "Mazda"], index=0)

# Model options filtered by Brand
model_options = []
if not df.empty and "Brand" in df.columns and "Model" in df.columns:
    model_options = sorted(df.loc[df["Brand"].astype(str) == str(Brand), "Model"].dropna().astype(str).unique().tolist())

if model_options:
    Model = st.selectbox("Model", model_options, index=0)
else:
    Model = st.text_input("Model", "Corolla")

Transmission_opts = safe_unique("Transmission")
Transmission = st.selectbox("Transmission", Transmission_opts if Transmission_opts else ["Automatic", "Manual", "Other"])

FuelType_opts = safe_unique("FuelType")
FuelType = st.selectbox("FuelType", FuelType_opts if FuelType_opts else ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])

UsedOrNew_opts = safe_unique("UsedOrNew")
UsedOrNew = st.selectbox("UsedOrNew", UsedOrNew_opts if UsedOrNew_opts else ["Used", "New"])

DriveType_opts = safe_unique("DriveType")
DriveType = st.selectbox("DriveType", DriveType_opts if DriveType_opts else ["FWD", "RWD", "AWD", "4WD", "Other"])

BodyType_opts = safe_unique("BodyType")
BodyType = st.selectbox("BodyType", BodyType_opts if BodyType_opts else ["Sedan", "Hatchback", "SUV", "Wagon"])

# Car/Suv + Location can be large (searchable)
CarSuv_opts = safe_unique("Car/Suv")
Car_Suv = st.selectbox("Car/Suv", CarSuv_opts if CarSuv_opts else ["SUV", "Sedan", "Hatchback"])

Location_opts = safe_unique("Location")
Location = st.selectbox("Location", Location_opts if Location_opts else ["NSW", "VIC", "QLD"], index=0)

# Strings (free text but required)
Title = st.text_input("Title (listing title / description)", "2018 Toyota Corolla Auto Hatch")
Engine = st.text_input("Engine", "2.0L")
ColourExtInt = st.text_input("ColourExtInt (e.g., 'White / Black')", "White / Black")

# Year validation using dataset min/max
Year = st.number_input("Year", min_value=int(year_min), max_value=int(year_max), value=min(2018, int(year_max)), step=1)

# Doors / Seats / Cylinders use dropdowns from CSV so format matches training
Doors_opts = safe_unique("Doors")
Doors = st.selectbox("Doors", Doors_opts if Doors_opts else [" 4 Doors", " 5 Doors", " 2 Doors"])

Seats_opts = safe_unique("Seats")
Seats = st.selectbox("Seats", Seats_opts if Seats_opts else [" 5 Seats", " 7 Seats", " 2 Seats"])

Cyl_opts = safe_unique("CylindersinEngine")
CylindersinEngine = st.selectbox("CylindersinEngine", Cyl_opts if Cyl_opts else ["4 cyl", "6 cyl", "8 cyl"])

# Kilometres: numeric input -> cast to training-like string (often plain digits)
Kilometres_num = st.number_input("Kilometres", min_value=0, value=60000, step=1000)
Kilometres = str(int(Kilometres_num))

# FuelConsumption: allow unknown
fuel_unknown = st.checkbox("FuelConsumption unknown", value=False)
if fuel_unknown:
    FuelConsumption = "-"  # matches training missing marker
else:
    FuelConsumption_num = st.number_input(
        "FuelConsumption (L / 100 km)",
        min_value=0.0,
        max_value=50.0,
        value=7.5,
        step=0.1,
    )
    # Format similar to dataset: "8.7 L / 100 km" or "11 L / 100 km"
    if abs(FuelConsumption_num - round(FuelConsumption_num)) < 1e-9:
        FuelConsumption = f"{int(round(FuelConsumption_num))} L / 100 km"
    else:
        FuelConsumption = f"{FuelConsumption_num:.1f} L / 100 km"

# ----------------------------
# Validation
# ----------------------------
errors = []
warnings = []

# Required text fields
if not str(Title).strip():
    errors.append("Title is required.")
if not str(Engine).strip():
    errors.append("Engine is required.")
if not str(ColourExtInt).strip():
    errors.append("ColourExtInt is required.")

# Sanity checks (warnings, not hard stops)
if Kilometres_num > km_p99:
    warnings.append(f"Kilometres looks very high vs dataset (>{int(km_p99):,}). Prediction may be less reliable.")

if not fuel_unknown:
    if FuelConsumption_num > fc_p99:
        warnings.append(f"FuelConsumption looks very high vs dataset (>{fc_p99:.1f}). Prediction may be less reliable.")

# Show validation messages
if errors:
    st.error("Please fix the following before predicting:\n- " + "\n- ".join(errors))

if warnings:
    st.warning("Notes:\n- " + "\n- ".join(warnings))

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict", disabled=bool(errors)):
    X = pd.DataFrame([{
        "Brand": Brand,
        "FuelType": FuelType,
        "Year": Year,
        "Transmission": Transmission,
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
        "Title": Title,
    }])

    # Extra guard: ensure columns match expected set
    missing = set(EXPECTED_FEATURES) - set(X.columns)
    extra = set(X.columns) - set(EXPECTED_FEATURES)
    if missing:
        st.error(f"Internal error: missing columns: {missing}")
        st.stop()
    if extra:
        st.error(f"Internal error: unexpected columns: {extra}")
        st.stop()

    try:
        pred = pipe.predict(X)[0]
        st.success(f"Estimated price: ${pred:,.0f}")

        with st.expander("Debug (sent to model)"):
            st.write("Columns:", list(X.columns))
            st.dataframe(X)

    except Exception as e:
        st.error("Prediction failed. See details below:")
        st.code(str(e))
        st.write("Debug — DataFrame sent to model:")
        st.dataframe(X)

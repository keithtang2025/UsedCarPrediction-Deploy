from pathlib import Path
import joblib
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

@st.cache_resource
def load_model():
    model_url = st.secrets["MODEL_URL"]  # set in Streamlit secrets

    if not MODEL_PATH.exists():
        with st.spinner("Downloading model... (first run only)"):
            r = requests.get(model_url, stream=True, timeout=300)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

    return joblib.load(MODEL_PATH)

pipe = load_model()

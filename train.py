# train.py
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# --- Paths (robust on GitHub Actions / Streamlit Cloud) ---
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "AustralianVehiclePrices.csv"

# Put your intended target here; code will also auto-detect if not found
TARGET_CANDIDATES = ["Price", "price", "PRICE", "VehiclePrice", "vehicle_price"]


def detect_target_column(df: pd.DataFrame) -> str:
    # 1) direct matches
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c

    # 2) fuzzy-ish: any column name containing "price"
    lower_map = {col.lower(): col for col in df.columns}
    if "price" in lower_map:
        return lower_map["price"]

    for col in df.columns:
        if "price" in col.lower():
            return col

    raise ValueError(
        f"Target column not found. Columns are: {df.columns.tolist()[:50]} ... "
        f"(showing first 50). Please set TARGET_CANDIDATES correctly."
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {DATA_PATH}. Files in repo: {[p.name for p in ROOT.iterdir()]}")

    df = pd.read_csv(DATA_PATH)

    target_col = detect_target_column(df)

    # Clean target
    y_raw = df[target_col]
    y = pd.to_numeric(y_raw, errors="coerce")
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep]

    # Features
    X = df.drop(columns=[target_col])

    # Basic split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify column types
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Pipelines with imputation (prevents many runtime crashes)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop"
    )

    # Keep model reasonable size for committing to GitHub
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("Target column:", target_col)
    print("Train rows:", len(X_train), "Test rows:", len(X_test))
    print("Categorical cols:", len(cat_cols), "Numeric cols:", len(num_cols))
    print("MAE:", mae)

    out_dir = ROOT / "model"
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "model.pkl"
    joblib.dump(pipe, model_path)
    print("Saved to:", model_path)


if __name__ == "__main__":
    main()

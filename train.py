# train.py
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "AustralianVehiclePrices.csv"  # adjust if needed
TARGET = "Price"                           # adjust to your target column

def main():
    df = pd.read_csv(DATA_PATH)

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Basic split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Infer columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("MAE:", mae)

    out_dir = Path("model")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(pipe, out_dir / "model.pkl")
    print("Saved to model/model.pkl")

if __name__ == "__main__":
    main()

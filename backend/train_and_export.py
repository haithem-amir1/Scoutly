# backend/train_and_export.py

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ----- Paths -----
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

INPUT_CSV = DATA_PROCESSED / "players_merged.csv"
OUTPUT_CSV = DATA_PROCESSED / "players_with_valuations.csv"
OUTPUT_JSON = DATA_PROCESSED / "players_with_valuations.json"
MODEL_PATH = MODELS_DIR / "market_value_model.joblib"

CURRENT_SEASON_YEAR = 2024
UNDER_Q = 0.8   # top 20% undervalued
OVER_Q = 0.2    # bottom 20% overvalued


# ----- Feature Engineering -----

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates engineered features:
      - goals_per90, assists_per90
      - age_bucket
      - contract_years_remaining
    """

    def __init__(self, current_season_year: int = CURRENT_SEASON_YEAR):
        self.current_season_year = current_season_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        minutes = X["minutes"].replace(0, np.nan)

        X["goals_per90"] = X["goals"] * 90 / minutes
        X["assists_per90"] = X["assists"] * 90 / minutes

        X["goals_per90"] = X["goals_per90"].fillna(0.0)
        X["assists_per90"] = X["assists_per90"].fillna(0.0)

        # Age buckets
        def bucket_age(a):
            if pd.isna(a):
                return "unknown"
            a = int(a)
            if a < 23:
                return "young"
            elif a <= 28:
                return "peak"
            else:
                return "veteran"

        X["age_bucket"] = X["age"].apply(bucket_age)

        # Contract years remaining (we don't really have this data, so it's NaN,
        # but the imputer/transformer will just skip as the warning says)
        X["contract_years_remaining"] = (
            X["contract_expires_year"].astype("float") - self.current_season_year
        )

        return X


def get_raw_feature_cols():
    """Columns before feature engineering."""
    return [
        "age",
        "minutes",
        "goals",
        "assists",
        "contract_expires_year",
        "club",
        "position",
    ]


def build_preprocessor():
    numeric_features = [
        "age",
        "minutes",
        "goals",
        "assists",
        "contract_expires_year",
        "goals_per90",
        "assists_per90",
        "contract_years_remaining",
    ]
    categorical_features = ["club", "position", "age_bucket"]

    num_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # older sklearn: use sparse_output instead of sparse
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return Pipeline(
        steps=[
            ("features", FeatureCreator()),
            (
                "cols",
                ColumnTransformer(
                    transformers=[
                        ("num", num_trans, numeric_features),
                        ("cat", cat_trans, categorical_features),
                    ]
                ),
            ),
        ]
    )


# ----- Model Training -----

def train_models(df: pd.DataFrame):
    feature_cols = get_raw_feature_cols()
    X = df[feature_cols].copy()
    y = df["market_value_million_eur"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    models_def = {
        "lasso": Lasso(alpha=0.1, max_iter=10000, random_state=42),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "rf": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
    }

    results = []
    fitted = {}

    for name, est in models_def.items():
        print(f"Training {name}...")
        pipe = Pipeline(
            steps=[("preprocess", preprocessor), ("model", est)]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        # older sklearn: no squared=False, so do sqrt manually
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted[name] = pipe

    metrics = pd.DataFrame(results).sort_values("RMSE")
    print("\nModel performance:")
    print(metrics)

    best_name = metrics.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

    return fitted[best_name]


# ----- Add predictions & labels -----

def add_valuations(df: pd.DataFrame, model):
    feature_cols = get_raw_feature_cols()
    df = df.copy()

    df["predicted_value_million_eur"] = model.predict(df[feature_cols])
    df["value_diff"] = (
        df["predicted_value_million_eur"] - df["market_value_million_eur"]
    )

    hi = df["value_diff"].quantile(UNDER_Q)
    lo = df["value_diff"].quantile(OVER_Q)

    def label(v):
        if v >= hi:
            return "undervalued"
        elif v <= lo:
            return "overvalued"
        else:
            return "fair"

    df["valuation_label"] = df["value_diff"].apply(label)
    return df


# ----- Main script -----

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # drop rows where target is missing
    df = df.dropna(subset=["market_value_million_eur"])

    # Train model + choose best
    model = train_models(df)

    # Generate predictions
    df_val = add_valuations(df, model)

    # Save CSV
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_val.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")

    # Save JSON for frontend
    records = df_val.to_dict(orient="records")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

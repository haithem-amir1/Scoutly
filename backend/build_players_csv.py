# backend/build_players_csv.py

import pandas as pd
from pathlib import Path

# Paths relative to the repo root
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_FILES = [
    RAW_DIR / "transfermarkt_fbref_201718.csv",
    RAW_DIR / "transfermarkt_fbref_201819.csv",
    RAW_DIR / "transfermarkt_fbref_201920.csv",
]

OUTPUT = PROCESSED_DIR / "players_merged.csv"


def load_and_clean(path: Path) -> pd.DataFrame:
    # Kaggle files use semicolons
    df = pd.read_csv(path, sep=";")

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Columns we care about
    keep = [
        "player",
        "squad",
        "age",
        "position",
        "minutes",
        "goals",
        "assists",
        "value",  # market value in euros
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Convert numeric columns
    for col in ["age", "minutes", "goals", "assists", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # We don't have real contract expiry yet
    df["contract_expires_year"] = pd.NA

    # Convert market value from euros to MILLIONS of euros
    df["market_value_million_eur"] = df["value"] / 1_000_000

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in RAW_FILES:
        print(f"Loading {f}...")
        frames.append(load_and_clean(f))

    df = pd.concat(frames, ignore_index=True)

    # Rename to standard schema
    df = df.rename(
        columns={
            "player": "player_name",
            "squad": "club",
            "value": "market_value_eur",
        }
    )

    # Final column order
    cols = [
        "player_name",
        "club",
        "age",
        "position",
        "minutes",
        "goals",
        "assists",
        "contract_expires_year",
        "market_value_million_eur",
    ]
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv(OUTPUT, index=False)
    print(f"Saved merged players dataset to {OUTPUT}")
    print(df.head())


if __name__ == "__main__":
    main()

"""
Phase 2 — Step 1: Load and validate the Phase 1 cleaned CSV.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import CLEANED_CSV

REQUIRED_COLS = {"title", "description", "transcript", "views", "duration"}


def load(path: str = CLEANED_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: Cleaned CSV not found at:\n  {path}")
        print("Run Phase 1 first:  py -3.12 src/phase1_data_cleaning.py")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from Phase 1 output.")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"ERROR: Missing expected columns: {missing}")
        sys.exit(1)

    for col in ["title", "description", "transcript"]:
        df[col] = df[col].fillna("").astype(str)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load()
    print(df[["title", "description"]].head(3).to_string())

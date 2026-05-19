"""
Phase 1: Data Collection & Cleaning
TED Talk Sentiment-Based Recommendation System
"""

import ast
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data", "2020-05-01", "ted_talks_en.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_ted_talks.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# -- 1. Load ------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# -- 2. EDA -------------------------------------------------------------------

def run_eda(df):
    print("\n-- Missing values (%) --")
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(missing[missing > 0].to_string())

    print("\n-- Numeric summary --")
    print(df[["views", "duration", "comments"]].describe().to_string())

    # Views + duration distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["views"].dropna(), bins=60, log=True, color="steelblue", edgecolor="white")
    axes[0].set_title("Views distribution (log scale)")
    axes[0].set_xlabel("Views")
    axes[1].hist(df["duration"].dropna(), bins=40, color="coral", edgecolor="white")
    axes[1].set_title("Talk duration (seconds)")
    axes[1].set_xlabel("Duration")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "distributions.png"), dpi=100)
    plt.close()
    print(f"\nSaved -> {PLOTS_DIR}/distributions.png")

    # Top topics
    topics_series = df["topics"].dropna().apply(_safe_parse_list)
    all_topics = [t for sublist in topics_series for t in sublist]
    top_topics = pd.Series(all_topics).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_topics.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 topics")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_topics.png"), dpi=100)
    plt.close()
    print(f"Saved -> {PLOTS_DIR}/top_topics.png")

    # Top occupations
    occ_series = df["occupations"].dropna().apply(_parse_dict_values)
    all_occ = [o for sublist in occ_series for o in sublist]
    top_occ = pd.Series(all_occ).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_occ.plot(kind="barh", ax=ax, color="coral")
    ax.set_title("Top 20 speaker occupations")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_occupations.png"), dpi=100)
    plt.close()
    print(f"Saved -> {PLOTS_DIR}/top_occupations.png")


# -- 3. Parsing helpers -------------------------------------------------------

def _safe_parse_list(val):
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _parse_dict_values(val):
    if pd.isna(val):
        return []
    try:
        d = ast.literal_eval(val)
        if isinstance(d, dict):
            flat = []
            for v in d.values():
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
            return flat
        return []
    except Exception:
        return []


# -- 4. Clean -----------------------------------------------------------------

def clean(df):
    df = df.copy()

    # Keep only English talks
    df = df[df["native_lang"] == "en"].copy()
    print(f"\nAfter English filter: {len(df):,} rows")

    # Drop rows missing transcript
    before = len(df)
    df = df.dropna(subset=["transcript"])
    print(f"Dropped {before - len(df)} rows with missing transcripts -> {len(df):,} remain")

    # Fill missing text fields
    for col in ["description", "title", "topics", "occupations"]:
        df[col] = df[col].fillna("")

    # Parse date columns
    for col in ["recorded_date", "published_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Cap views at 99th percentile (keep raw views too)
    p99 = df["views"].quantile(0.99)
    df["views_capped"] = df["views"].clip(upper=p99)
    print(f"Views 99th percentile cap: {p99:,.0f}")

    # Drop extreme duration outliers
    before = len(df)
    df = df[(df["duration"] >= 60) & (df["duration"] <= 6000)]
    print(f"Dropped {before - len(df)} rows with extreme duration -> {len(df):,} remain")

    # Parse structured columns
    df["topics_list"] = df["topics"].apply(_safe_parse_list)
    df["occupations_list"] = df["occupations"].apply(_parse_dict_values)

    return df.reset_index(drop=True)


# -- 5. Feature engineering ---------------------------------------------------

def build_features(df):
    df = df.copy()

    # TF-IDF: titles
    tfidf_title = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    title_matrix = tfidf_title.fit_transform(df["title"].fillna(""))
    title_df = pd.DataFrame(
        title_matrix.toarray(),
        columns=[f"tfidf_title_{v}" for v in tfidf_title.get_feature_names_out()],
        index=df.index,
    )

    # TF-IDF: descriptions
    tfidf_desc = TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2))
    desc_matrix = tfidf_desc.fit_transform(df["description"].fillna(""))
    desc_df = pd.DataFrame(
        desc_matrix.toarray(),
        columns=[f"tfidf_desc_{v}" for v in tfidf_desc.get_feature_names_out()],
        index=df.index,
    )

    # Multi-hot topics
    mlb_topics = MultiLabelBinarizer()
    topics_encoded = mlb_topics.fit_transform(df["topics_list"])
    topics_df = pd.DataFrame(
        topics_encoded,
        columns=[f"topic_{t}" for t in mlb_topics.classes_],
        index=df.index,
    )

    # Multi-hot occupations
    mlb_occ = MultiLabelBinarizer()
    occ_encoded = mlb_occ.fit_transform(df["occupations_list"])
    occ_df = pd.DataFrame(
        occ_encoded,
        columns=[f"occ_{o.replace(' ', '_')}" for o in mlb_occ.classes_],
        index=df.index,
    )

    print(f"\nFeature dimensions:")
    print(f"  TF-IDF title:        {title_df.shape[1]}")
    print(f"  TF-IDF description:  {desc_df.shape[1]}")
    print(f"  Topic flags:         {topics_df.shape[1]}")
    print(f"  Occupation flags:    {occ_df.shape[1]}")

    df = pd.concat([df, topics_df, occ_df], axis=1)
    return df, title_df, desc_df


# -- 6. Save ------------------------------------------------------------------

def save(df, path):
    drop_cols = ["topics_list", "occupations_list"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.to_csv(path, index=False)
    print(f"\nCleaned dataset saved -> {path}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 1: Data Collection & Cleaning")
    print("=" * 60)

    df = load_data(DATA_PATH)
    run_eda(df)
    df = clean(df)
    df, title_tfidf, desc_tfidf = build_features(df)
    save(df, OUTPUT_PATH)

    print("\nPhase 1 complete.")
    print(f"  Cleaned CSV:  {OUTPUT_PATH}")
    print(f"  EDA plots:    {PLOTS_DIR}/")


if __name__ == "__main__":
    main()

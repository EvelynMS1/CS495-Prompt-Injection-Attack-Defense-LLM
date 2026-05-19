"""
Phase 2 — Step 3: Sentiment analysis on TED Talk transcripts.

Models:
  - DistilBERT (distilbert-base-uncased-finetuned-sst-2-english): POSITIVE/NEGATIVE polarity + score
  - RoBERTa GoEmotions (SamLowe/roberta-base-go_emotions): 28 labels collapsed to 7 emotions
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from config import (
    EMOTION_COLS,
    EMOTION_MODEL,
    GO_MAP,
    POLARITY_MODEL,
    TRANSCRIPT_CHARS,
)


# -- Model loaders ------------------------------------------------------------

def load_polarity_pipe():
    print(f"Loading DistilBERT polarity model: {POLARITY_MODEL} ...")
    return pipeline(
        "sentiment-analysis",
        model=POLARITY_MODEL,
        truncation=True,
        max_length=512,
    )


def load_emotion_pipe():
    print(f"Loading RoBERTa GoEmotions model: {EMOTION_MODEL} ...")
    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        truncation=True,
        max_length=512,
    )


# -- Inference helpers --------------------------------------------------------

def _polarity(pipe, text: str) -> tuple[str, float]:
    text = str(text).strip()
    if not text:
        return "NEUTRAL", 0.5
    result = pipe(text[:TRANSCRIPT_CHARS])[0]
    label = result["label"]
    score = result["score"] if label == "POSITIVE" else round(1.0 - result["score"], 4)
    return label, round(score, 4)


def _emotions(pipe, text: str) -> dict[str, float]:
    text = str(text).strip()
    if not text:
        return {e: 0.0 for e in EMOTION_COLS}

    results = pipe(text[:TRANSCRIPT_CHARS])[0]

    aggregated = {e: 0.0 for e in EMOTION_COLS}
    for r in results:
        broad = GO_MAP.get(r["label"], "neutral")
        aggregated[broad] = round(aggregated[broad] + r["score"], 4)

    total = sum(aggregated.values()) or 1.0
    return {k: round(v / total, 4) for k, v in aggregated.items()}


# -- Main enrichment function -------------------------------------------------

def enrich_with_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    polarity_pipe = load_polarity_pipe()
    emotion_pipe  = load_emotion_pipe()

    print(f"\nRunning sentiment analysis on {len(df):,} talks ...")
    print(f"  Using first {TRANSCRIPT_CHARS} chars of each transcript.")

    polarities, polarity_scores, emotion_records = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment + Emotions"):
        text = str(row.get("transcript", ""))

        label, score = _polarity(polarity_pipe, text)
        polarities.append(label)
        polarity_scores.append(score)

        emotion_records.append(_emotions(emotion_pipe, text))

    df = df.copy()
    df["polarity"]       = polarities
    df["polarity_score"] = polarity_scores

    emotion_df = pd.DataFrame(emotion_records, index=df.index).fillna(0.0)
    df = pd.concat([df, emotion_df], axis=1)

    print("\n-- Polarity distribution --")
    print(df["polarity"].value_counts().to_string())
    print("\n-- Mean emotion scores --")
    for col in EMOTION_COLS:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.4f}")

    return df


if __name__ == "__main__":
    from config import CLEANED_CSV

    sample = pd.read_csv(CLEANED_CSV, nrows=5)
    sample["transcript"] = sample["transcript"].fillna("").astype(str)
    result = enrich_with_sentiment(sample)
    print(result[["title", "polarity", "polarity_score"] + EMOTION_COLS].to_string())

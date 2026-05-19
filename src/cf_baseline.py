"""
CF Item Bias Baseline
=====================
Scores talks using: 0.7 × log-normalized popularity (views) + 0.3 × sentiment alignment.

This is a bias-only collaborative filtering model in the style of Koren (2009):
  score(i, u) = b_i + b_u
where b_i = log-normalized view count (item popularity bias) and b_u is replaced by
the same emotion-alignment function used in the main hybrid engine.

No semantic understanding of the query text is used — that is the point of the baseline.
It answers the question: "how much does semantic retrieval add over pure popularity + mood?"

Output interface is identical to RecommendationEngine.recommend() so results can be
compared directly in the UI without any changes to calling code.
"""

import os

import numpy as np
import pandas as pd

ENRICHED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "enriched_ted_talks.csv")
EMOTION_COLS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

_POPULARITY_WEIGHT = 0.7
_SENTIMENT_WEIGHT = 0.3


class CFBiasBaseline:
    """
    Item-bias CF baseline. query_text is intentionally ignored — ranking is driven
    only by global item popularity (log-normalized views) and emotion alignment.
    """

    def __init__(self, csv_path: str = ENRICHED_CSV):
        df = pd.read_csv(csv_path)

        # --- item bias: log-normalize view counts ---
        log_views = np.log1p(df["views"].fillna(0).clip(lower=0))
        v_min, v_max = log_views.min(), log_views.max()
        df["_pop_score"] = (log_views - v_min) / (v_max - v_min + 1e-9)

        # Handle both speaker column names that exist across pipeline versions
        speaker_col = "speaker_1" if "speaker_1" in df.columns else "speaker"
        df["_speaker"] = df[speaker_col].fillna("").astype(str)

        self._df = df
        print(
            f"CFBiasBaseline ready: {len(df):,} talks loaded. "
            f"Popularity range [{log_views.min():.2f}, {log_views.max():.2f}] (log-views)."
        )

    def _sentiment_alignment(self, row: pd.Series, user_emotions: dict) -> float:
        """Identical formula to RecommendationEngine._sentiment_alignment."""
        if not user_emotions:
            return 1.0
        scores = [
            1.0 - abs(user_emotions.get(e, 0.0) - float(row.get(e, 0.0) or 0.0))
            for e in EMOTION_COLS
        ]
        return sum(scores) / len(scores)

    def recommend(self, query_text: str, user_emotions: dict = None, top_k: int = 5):
        """
        Same signature as RecommendationEngine.recommend().
        query_text is accepted but not used (by design — this is the baseline).
        Returns a list of dicts with the same keys as the main engine, plus
        'popularity_score' in place of 'cosine_sim'.
        """
        emo = user_emotions or {}
        records = []

        for _, row in self._df.iterrows():
            pop = float(row["_pop_score"])
            sent = self._sentiment_alignment(row, emo)
            cf_score = _POPULARITY_WEIGHT * pop + _SENTIMENT_WEIGHT * sent

            records.append({
                "title": str(row.get("title", "")),
                "speaker": row["_speaker"],
                "url": str(row.get("url", "")),
                "description": str(row.get("description", "")),
                "views": int(row.get("views", 0) or 0),
                "duration": int(row.get("duration", 0) or 0),
                "polarity": str(row.get("polarity", "")),
                "polarity_score": round(float(row.get("polarity_score", 0.5) or 0.5), 3),
                "emotions": {e: round(float(row.get(e, 0.0) or 0.0), 3) for e in EMOTION_COLS},
                "popularity_score": round(pop, 4),
                "sentiment_alignment": round(sent, 4),
                "combined_score": round(cf_score, 4),
            })

        records.sort(key=lambda x: x["combined_score"], reverse=True)
        return records[:top_k]

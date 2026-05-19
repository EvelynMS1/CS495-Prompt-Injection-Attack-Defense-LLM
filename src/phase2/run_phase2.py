"""
Phase 2 orchestrator — run this file to execute the full Phase 2 pipeline.

    py -3.12 src/phase2/run_phase2.py

Steps:
    1. Load cleaned CSV from Phase 1
    2. Sentiment analysis (DistilBERT polarity + RoBERTa GoEmotions)   [skipped if enriched CSV exists]
    3. BERT content embeddings (bert-base-uncased, 768-dim, mean pooled)
    4. Store in ChromaDB (collection: ted_talks_bert)

Outputs:
    data/enriched_ted_talks_bert.csv
    data/chromadb_bert/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import CLEANED_CSV, EMOTION_COLS, ENRICHED_CSV
from data_loader import load
from bert_embeddings import BERTEmbedder, generate_embeddings
from sentiment_analysis import enrich_with_sentiment
from vector_store import load_into_chromadb


def main():
    print("=" * 60)
    print("Phase 2: Sentiment Analysis + BERT Embeddings + ChromaDB")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1 – Load                                                        #
    # ------------------------------------------------------------------ #
    df = load()

    # ------------------------------------------------------------------ #
    # Step 2 – Sentiment analysis (expensive; skip if already done)        #
    # ------------------------------------------------------------------ #
    if os.path.exists(ENRICHED_CSV):
        candidate = pd.read_csv(ENRICHED_CSV)
        sentiment_done = "polarity" in candidate.columns and all(
            e in candidate.columns for e in EMOTION_COLS
        )
        if sentiment_done:
            print(f"\nEnriched CSV already exists ({len(candidate):,} rows). Skipping sentiment step.")
            df = candidate
        else:
            print("\nEnriched CSV found but missing sentiment columns — re-running sentiment.")
            df = enrich_with_sentiment(df)
            df.to_csv(ENRICHED_CSV, index=False)
            print(f"Saved -> {ENRICHED_CSV}")
    else:
        df = enrich_with_sentiment(df)
        df.to_csv(ENRICHED_CSV, index=False)
        print(f"\nEnriched CSV saved -> {ENRICHED_CSV}")

    # ------------------------------------------------------------------ #
    # Step 3 – BERT embeddings                                             #
    # ------------------------------------------------------------------ #
    embedder   = BERTEmbedder()
    embeddings = generate_embeddings(df, embedder)

    # ------------------------------------------------------------------ #
    # Step 4 – ChromaDB                                                    #
    # ------------------------------------------------------------------ #
    load_into_chromadb(df, embeddings)

    print("\n" + "=" * 60)
    print("Phase 2 complete.")
    print(f"  Enriched CSV : {ENRICHED_CSV}")
    print(f"  ChromaDB     : data/chromadb_bert/")
    print("=" * 60)


if __name__ == "__main__":
    main()

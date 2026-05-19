"""
Phase 2: Sentiment Analysis + Embeddings + ChromaDB
Models: DistilBERT (polarity), RoBERTa GoEmotions (7 emotions), all-MiniLM-L6-v2 (embeddings)
"""

import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb

warnings.filterwarnings("ignore")

CLEANED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_ted_talks.csv")
ENRICHED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "enriched_ted_talks.csv")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# SamLowe model uses safetensors weights — compatible with any torch version
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Use first 1500 chars of transcript (~200-250 words) — fast on CPU, captures the talk's opening tone
TRANSCRIPT_CHARS = 1500

EMOTION_COLS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Map GoEmotions 28 labels -> 7 broad categories
_GO_MAP = {
    "joy": "joy", "amusement": "joy", "excitement": "joy", "gratitude": "joy",
    "admiration": "joy", "approval": "joy", "love": "joy", "pride": "joy",
    "relief": "joy", "optimism": "joy", "caring": "joy",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness", "remorse": "sadness",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
    "neutral": "neutral", "curiosity": "neutral", "desire": "neutral",
    "embarrassment": "neutral",
}


# -- Model loaders ------------------------------------------------------------

def load_sentiment_pipe():
    print("Loading DistilBERT sentiment model...")
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
    )


def load_emotion_pipe():
    print("Loading RoBERTa emotion model (GoEmotions)...")
    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        truncation=True,
        max_length=512,
    )


def load_embed_model():
    print("Loading sentence-transformer embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL)


# -- Inference helpers --------------------------------------------------------

def _get_polarity(pipe, text):
    if not text or not str(text).strip():
        return "NEUTRAL", 0.5
    result = pipe(str(text)[:TRANSCRIPT_CHARS])[0]
    label = result["label"]
    score = result["score"] if label == "POSITIVE" else 1.0 - result["score"]
    return label, round(score, 4)


def _get_emotions(pipe, text):
    if not text or not str(text).strip():
        return {e: 0.0 for e in EMOTION_COLS}
    results = pipe(str(text)[:TRANSCRIPT_CHARS])[0]
    # Aggregate 28 GoEmotions labels into 7 broad categories
    aggregated = {e: 0.0 for e in EMOTION_COLS}
    for r in results:
        broad = _GO_MAP.get(r["label"], "neutral")
        aggregated[broad] = round(aggregated[broad] + r["score"], 4)
    # Normalize so scores sum to ~1
    total = sum(aggregated.values()) or 1.0
    return {k: round(v / total, 4) for k, v in aggregated.items()}


# -- Phase 2a: Sentiment scoring ----------------------------------------------

def enrich_with_sentiment(df, sentiment_pipe, emotion_pipe):
    print(f"\nRunning sentiment analysis on {len(df):,} talks...")
    print("(Using first 1500 chars of transcript per talk)")

    polarities, polarity_scores = [], []
    emotion_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment + Emotions"):
        text = str(row.get("transcript", ""))

        label, score = _get_polarity(sentiment_pipe, text)
        polarities.append(label)
        polarity_scores.append(score)

        emotions = _get_emotions(emotion_pipe, text)
        emotion_records.append(emotions)

    df = df.copy()
    df["polarity"] = polarities
    df["polarity_score"] = polarity_scores

    emotion_df = pd.DataFrame(emotion_records, index=df.index).fillna(0.0)
    df = pd.concat([df, emotion_df], axis=1)

    print("\n-- Sentiment distribution --")
    print(df["polarity"].value_counts().to_string())
    print("\n-- Average emotion scores --")
    for col in EMOTION_COLS:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.3f}")

    return df


# -- Phase 2b: Embeddings -----------------------------------------------------

def generate_embeddings(df, embed_model):
    print(f"\nGenerating embeddings for {len(df):,} talks...")
    # Embed title + description — concise and semantic
    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True)
    print(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings


# -- Phase 2c: Load ChromaDB --------------------------------------------------

def load_chromadb(df, embeddings):
    print(f"\nLoading {len(df):,} talks into ChromaDB at {CHROMA_DIR}...")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop and recreate to ensure clean state
    try:
        client.delete_collection("ted_talks")
        print("Dropped existing 'ted_talks' collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name="ted_talks",
        metadata={"hnsw:space": "cosine"},
    )

    present_emotions = [c for c in EMOTION_COLS if c in df.columns]

    ids, docs, metas, embs = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        talk_id = str(int(row["talk_id"])) if pd.notna(row.get("talk_id")) else str(i)

        meta = {
            "talk_id": talk_id,
            "title": str(row.get("title", ""))[:500],
            "speaker": str(row.get("speaker_1", ""))[:200],
            "views": int(row["views"]) if pd.notna(row.get("views")) else 0,
            "duration": int(row["duration"]) if pd.notna(row.get("duration")) else 0,
            "url": str(row.get("url", ""))[:500],
            "description": str(row.get("description", ""))[:1000],
            "polarity": str(row.get("polarity", "NEUTRAL")),
            "polarity_score": float(row.get("polarity_score", 0.5)),
        }
        for ec in present_emotions:
            meta[ec] = float(row.get(ec, 0.0)) if pd.notna(row.get(ec)) else 0.0

        ids.append(talk_id)
        docs.append(f"{row.get('title', '')} {row.get('description', '')}")
        metas.append(meta)
        embs.append(embeddings[i].tolist())

    # Batch insert
    batch_size = 100
    for start in tqdm(range(0, len(ids), batch_size), desc="ChromaDB insert"):
        collection.add(
            ids=ids[start:start + batch_size],
            documents=docs[start:start + batch_size],
            metadatas=metas[start:start + batch_size],
            embeddings=embs[start:start + batch_size],
        )

    count = collection.count()
    print(f"ChromaDB 'ted_talks' collection loaded with {count:,} entries.")
    return collection


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 2: Sentiment Analysis & Embeddings")
    print("=" * 60)

    # If enriched CSV already exists with sentiment columns, skip re-running
    # the expensive sentiment inference (takes hours on CPU).
    if os.path.exists(ENRICHED_CSV):
        df = pd.read_csv(ENRICHED_CSV)
        if "polarity" in df.columns and "joy" in df.columns:
            print(f"Enriched CSV already exists ({len(df):,} rows). Skipping sentiment step.")
        else:
            if not os.path.exists(CLEANED_CSV):
                print(f"ERROR: Cleaned CSV not found at {CLEANED_CSV}")
                print("Run Phase 1 first: py -3.12 src/phase1_data_cleaning.py")
                raise SystemExit(1)
            df = pd.read_csv(CLEANED_CSV)
            print(f"Loaded {len(df):,} rows from Phase 1 output.")
            sentiment_pipe = load_sentiment_pipe()
            emotion_pipe = load_emotion_pipe()
            df = enrich_with_sentiment(df, sentiment_pipe, emotion_pipe)
            df.to_csv(ENRICHED_CSV, index=False)
            print(f"\nEnriched dataset saved -> {ENRICHED_CSV}")
    else:
        if not os.path.exists(CLEANED_CSV):
            print(f"ERROR: Cleaned CSV not found at {CLEANED_CSV}")
            print("Run Phase 1 first: py -3.12 src/phase1_data_cleaning.py")
            raise SystemExit(1)
        df = pd.read_csv(CLEANED_CSV)
        print(f"Loaded {len(df):,} rows from Phase 1 output.")
        sentiment_pipe = load_sentiment_pipe()
        emotion_pipe = load_emotion_pipe()
        df = enrich_with_sentiment(df, sentiment_pipe, emotion_pipe)
        df.to_csv(ENRICHED_CSV, index=False)
        print(f"\nEnriched dataset saved -> {ENRICHED_CSV}")

    embed_model = load_embed_model()
    embeddings = generate_embeddings(df, embed_model)

    load_chromadb(df, embeddings)

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()

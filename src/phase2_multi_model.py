"""
Phase 2: Sentiment Analysis + Embeddings + ChromaDB (Multi-Model Support)
Supports multiple embedding models for flexible configuration
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import torch

from model_config import (
    EMBEDDING_MODELS,
    SENTIMENT_MODELS,
    EMOTION_MODELS,
    DEFAULT_EMBEDDING,
    DEFAULT_SENTIMENT,
    DEFAULT_EMOTION,
    EMOTION_COLS,
    GO_EMOTIONS_MAP,
)

warnings.filterwarnings("ignore")

CLEANED_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_ted_talks.csv")
TRANSCRIPT_CHARS = 1500


def load_sentiment_pipe(model_key):
    """Load sentiment analysis model."""
    model_config = SENTIMENT_MODELS[model_key]
    print(f"Loading sentiment model: {model_key}...")
    return pipeline(
        model_config["task"],
        model=model_config["model_name"],
        truncation=True,
        max_length=512,
    )


def load_emotion_pipe(model_key):
    """Load emotion detection model."""
    model_config = EMOTION_MODELS[model_key]
    print(f"Loading emotion model: {model_key}...")
    return pipeline(
        "text-classification",
        model=model_config["model_name"],
        top_k=None,
        truncation=True,
        max_length=512,
    ), model_config


def load_embed_model(model_key):
    """Load embedding model."""
    model_config = EMBEDDING_MODELS[model_key]
    model_name = model_config["model_name"]
    print(f"Loading embedding model: {model_key}...")
    print(f"  Model: {model_name}")
    print(f"  Dimensions: {model_config['dimensions']}")

    if model_name.startswith("sentence-transformers/"):
        return SentenceTransformer(model_name), True
    else:
        # Raw BERT-style model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return (model, tokenizer, device), False


def _get_polarity(pipe, text):
    """Extract sentiment polarity."""
    if not text or not str(text).strip():
        return "NEUTRAL", 0.5
    result = pipe(str(text)[:TRANSCRIPT_CHARS])[0]
    label = result["label"]
    score = result["score"] if label in ["POSITIVE", "positive"] else 1.0 - result["score"]
    return label.upper(), round(score, 4)


def _get_emotions(pipe, text, emotion_config):
    """Extract emotion scores with support for different models."""
    if not text or not str(text).strip():
        return {e: 0.0 for e in EMOTION_COLS}

    results = pipe(str(text)[:TRANSCRIPT_CHARS])[0]

    # If using GoEmotions (28 labels), aggregate to 7
    if emotion_config["num_emotions"] == 28:
        aggregated = {e: 0.0 for e in EMOTION_COLS}
        for r in results:
            broad = GO_EMOTIONS_MAP.get(r["label"], "neutral")
            aggregated[broad] = round(aggregated[broad] + r["score"], 4)
        # Normalize
        total = sum(aggregated.values()) or 1.0
        return {k: round(v / total, 4) for k, v in aggregated.items()}
    else:
        # Already 7 emotions
        return {r["label"]: round(r["score"], 4) for r in results}


def enrich_with_sentiment(df, sentiment_pipe, emotion_pipe, emotion_config):
    """Add sentiment and emotion columns to dataframe."""
    print(f"\nRunning sentiment analysis on {len(df):,} talks...")
    print("(Using first 1500 chars of transcript per talk)")

    polarities, polarity_scores = [], []
    emotion_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment + Emotions"):
        text = str(row.get("transcript", ""))

        label, score = _get_polarity(sentiment_pipe, text)
        polarities.append(label)
        polarity_scores.append(score)

        emotions = _get_emotions(emotion_pipe, text, emotion_config)
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


def generate_embeddings(df, embed_model_data, is_sentence_transformer):
    """Generate embeddings for all talks."""
    print(f"\nGenerating embeddings for {len(df):,} talks...")
    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()

    if is_sentence_transformer:
        embeddings = embed_model_data.encode(texts, batch_size=64, show_progress_bar=True)
    else:
        # Manual BERT encoding
        model, tokenizer, device = embed_model_data
        embeddings = []

        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                # Normalize
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)

    print(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings


def load_chromadb(df, embeddings, model_key):
    """Load embeddings into ChromaDB."""
    model_config = EMBEDDING_MODELS[model_key]
    chroma_dir = os.path.join(os.path.dirname(__file__), "..", "data", model_config["chroma_dir"])

    print(f"\nLoading {len(df):,} talks into ChromaDB at {chroma_dir}...")
    os.makedirs(chroma_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)

    collection_name = model_config["collection_name"]

    # Drop and recreate
    try:
        client.delete_collection(collection_name)
        print(f"Dropped existing '{collection_name}' collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
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
    print(f"ChromaDB '{collection_name}' collection loaded with {count:,} entries.")
    return collection


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings with configurable models"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=list(EMBEDDING_MODELS.keys()),
        default=DEFAULT_EMBEDDING,
        help="Embedding model to use"
    )
    parser.add_argument(
        "--sentiment-model",
        type=str,
        choices=list(SENTIMENT_MODELS.keys()),
        default=DEFAULT_SENTIMENT,
        help="Sentiment analysis model to use"
    )
    parser.add_argument(
        "--emotion-model",
        type=str,
        choices=list(EMOTION_MODELS.keys()),
        default=DEFAULT_EMOTION,
        help="Emotion detection model to use"
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment analysis if enriched CSV exists"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2: Multi-Model Sentiment Analysis & Embeddings")
    print("=" * 60)
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Sentiment Model: {args.sentiment_model}")
    print(f"Emotion Model: {args.emotion_model}")
    print("=" * 60)

    # Determine enriched CSV path based on model
    model_config = EMBEDDING_MODELS[args.embedding_model]
    enriched_csv = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        f"enriched_ted_talks_{model_config['chroma_dir'].replace('chromadb_', '')}.csv"
    )

    # Load or create enriched data
    if args.skip_sentiment and os.path.exists(enriched_csv):
        df = pd.read_csv(enriched_csv)
        if "polarity" in df.columns and "joy" in df.columns:
            print(f"Enriched CSV already exists ({len(df):,} rows). Skipping sentiment step.")
        else:
            raise ValueError("Enriched CSV missing required columns. Re-run without --skip-sentiment")
    else:
        if not os.path.exists(CLEANED_CSV):
            print(f"ERROR: Cleaned CSV not found at {CLEANED_CSV}")
            print("Run Phase 1 first: py -3.12 src/phase1_data_cleaning.py")
            raise SystemExit(1)

        df = pd.read_csv(CLEANED_CSV)
        print(f"Loaded {len(df):,} rows from Phase 1 output.")

        sentiment_pipe = load_sentiment_pipe(args.sentiment_model)
        emotion_pipe, emotion_config = load_emotion_pipe(args.emotion_model)
        df = enrich_with_sentiment(df, sentiment_pipe, emotion_pipe, emotion_config)

        df.to_csv(enriched_csv, index=False)
        print(f"\nEnriched dataset saved -> {enriched_csv}")

    # Generate embeddings
    embed_model_data, is_sentence_transformer = load_embed_model(args.embedding_model)
    embeddings = generate_embeddings(df, embed_model_data, is_sentence_transformer)

    # Load to ChromaDB
    load_chromadb(df, embeddings, args.embedding_model)

    print("\nPhase 2 complete.")
    print(f"✅ Embeddings generated with {args.embedding_model}")
    print(f"✅ Ready to use in chatbot with this model selection")


if __name__ == "__main__":
    main()

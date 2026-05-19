"""
Phase 2 — Step 4: Store BERT embeddings + sentiment metadata in ChromaDB.

Collection: ted_talks_bert
Directory:  data/chromadb_bert/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from tqdm import tqdm
import chromadb

from config import CHROMA_BATCH, CHROMA_DIR, EMOTION_COLS


def load_into_chromadb(df: pd.DataFrame, embeddings: np.ndarray) -> chromadb.Collection:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    print(f"\nConnecting to ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection("ted_talks_bert")
        print("Dropped existing 'ted_talks_bert' collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name="ted_talks_bert",
        metadata={"hnsw:space": "cosine"},
    )

    present_emotions = [c for c in EMOTION_COLS if c in df.columns]
    ids, docs, metas, embs = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        talk_id = str(int(row["talk_id"])) if pd.notna(row.get("talk_id")) else str(i)

        meta = {
            "talk_id":       talk_id,
            "title":         str(row.get("title", ""))[:500],
            "speaker":       str(row.get("speaker_1", ""))[:200],
            "views":         int(row["views"])    if pd.notna(row.get("views"))    else 0,
            "duration":      int(row["duration"]) if pd.notna(row.get("duration")) else 0,
            "url":           str(row.get("url", ""))[:500],
            "description":   str(row.get("description", ""))[:1000],
            "polarity":      str(row.get("polarity", "NEUTRAL")),
            "polarity_score": float(row.get("polarity_score", 0.5)),
        }
        for ec in present_emotions:
            meta[ec] = float(row.get(ec, 0.0)) if pd.notna(row.get(ec)) else 0.0

        ids.append(talk_id)
        docs.append(f"{row.get('title', '')} {row.get('description', '')}")
        metas.append(meta)
        embs.append(embeddings[i].tolist())

    print(f"Inserting {len(ids):,} talks into ChromaDB ...")
    for start in tqdm(range(0, len(ids), CHROMA_BATCH), desc="ChromaDB insert"):
        end = start + CHROMA_BATCH
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embs[start:end],
        )

    count = collection.count()
    print(f"Collection 'ted_talks_bert' loaded with {count:,} entries.")
    return collection


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("ted_talks_bert")


if __name__ == "__main__":
    col = get_collection()
    print(f"Collection has {col.count():,} entries.")

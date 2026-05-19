"""
Phase 2 Demo — query the BERT-powered ChromaDB collection.

    py -3.12 src/phase2/demo.py                  # runs preset sample queries
    py -3.12 src/phase2/demo.py "your query here" # runs a single custom query

ChromaDB collection: ted_talks_bert  (data/chromadb_bert/)
Embedding model:     bert-base-uncased (768-dim, mean pooled)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
import torch
from transformers import AutoModel, AutoTokenizer

from config import BERT_MODEL, CHROMA_DIR, EMOTION_COLS

CONTENT_WEIGHT  = 0.7
SENTIMENT_WEIGHT = 0.3

SAMPLE_QUERIES = [
    ("I want to feel inspired about climate change",        {"joy": 0.6, "sadness": 0.2}),
    ("Something funny and uplifting about technology",      {"joy": 0.8, "neutral": 0.2}),
    ("I feel anxious and want to understand mental health", {"fear": 0.5, "sadness": 0.4}),
    ("Curious about the universe and space exploration",    {"neutral": 0.5, "joy": 0.4}),
]


# -- BERT query encoder -------------------------------------------------------

class QueryEncoder:
    def __init__(self):
        print(f"Loading BERT encoder: {BERT_MODEL} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.model     = AutoModel.from_pretrained(BERT_MODEL)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.model(**encoded)
        mask   = encoded["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
        summed = torch.sum(output.last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (summed / counts).squeeze().tolist()


# -- Scoring ------------------------------------------------------------------

def _sentiment_alignment(meta: dict, user_emotions: dict) -> float:
    if not user_emotions:
        return 1.0
    scores = [1.0 - abs(float(meta.get(e, 0.0)) - v) for e, v in user_emotions.items()]
    return sum(scores) / len(scores)


def _fmt_duration(seconds) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# -- Search -------------------------------------------------------------------

def search(collection, encoder, query_text: str, user_emotions: dict = None, top_k: int = 5):
    vec = encoder.encode(query_text)

    n_candidates = min(top_k * 6, collection.count())
    results = collection.query(
        query_embeddings=[vec],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for i, meta in enumerate(results["metadatas"][0]):
        cosine_sim      = max(0.0, 1.0 - results["distances"][0][i])
        sentiment_score = _sentiment_alignment(meta, user_emotions or {})
        combined        = CONTENT_WEIGHT * cosine_sim + SENTIMENT_WEIGHT * sentiment_score
        candidates.append((combined, cosine_sim, sentiment_score, meta))

    candidates.sort(reverse=True)
    return candidates[:top_k]


# -- Display ------------------------------------------------------------------

def print_results(query_text: str, user_emotions: dict, results: list):
    print(f"\n{'='*65}")
    print(f"  Query : {query_text}")
    if user_emotions:
        mood_str = "  ".join(f"{k}={v}" for k, v in user_emotions.items())
        print(f"  Mood  : {mood_str}")
    print(f"{'='*65}")

    for rank, (combined, cosine, sentiment, meta) in enumerate(results, 1):
        dominant = max(EMOTION_COLS, key=lambda e: float(meta.get(e, 0.0)))
        print(f"\n  {rank}. {meta.get('title', 'N/A')}")
        print(f"     Speaker  : {meta.get('speaker', 'N/A')}")
        print(f"     Duration : {_fmt_duration(meta.get('duration', 0))}  |  "
              f"Views: {int(meta.get('views', 0)):,}")
        print(f"     Polarity : {meta.get('polarity', 'N/A')} "
              f"({float(meta.get('polarity_score', 0)):.2f})  |  "
              f"Dominant emotion: {dominant}")
        print(f"     Score    : {combined:.4f}  "
              f"(content={cosine:.4f}, sentiment={sentiment:.4f})")
        print(f"     URL      : {meta.get('url', 'N/A')}")


# -- Main ---------------------------------------------------------------------

def main():
    if not os.path.exists(CHROMA_DIR):
        print(f"ERROR: ChromaDB not found at {CHROMA_DIR}")
        print("Run Phase 2 first:  py -3.12 src/phase2/run_phase2.py")
        sys.exit(1)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("ted_talks_bert")
    print(f"Connected — {collection.count():,} talks in 'ted_talks_bert'.")

    encoder = QueryEncoder()

    # Custom query from command line
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = search(collection, encoder, query, top_k=5)
        print_results(query, {}, results)
        return

    # Preset sample queries
    for query_text, emotions in SAMPLE_QUERIES:
        results = search(collection, encoder, query_text, emotions, top_k=3)
        print_results(query_text, emotions, results)

    print(f"\n{'='*65}")
    print("Demo complete.")
    print(f"  Tip: py -3.12 src/phase2/demo.py \"your own query here\"")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

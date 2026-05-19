"""
Phase 3: Recommendation Engine
Hybrid ranking: cosine similarity (content) + sentiment alignment (emotion match)
Supports multiple embedding models via dynamic configuration
"""

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch

from model_config import EMBEDDING_MODELS, DEFAULT_EMBEDDING, EMOTION_COLS

CONTENT_WEIGHT = 0.7
SENTIMENT_WEIGHT = 0.3


class RecommendationEngine:
    def __init__(self, embedding_model_key=None):
        """
        Initialize recommendation engine with specified embedding model.

        Args:
            embedding_model_key: Key from EMBEDDING_MODELS dict, or None for default
        """
        if embedding_model_key is None:
            embedding_model_key = DEFAULT_EMBEDDING

        if embedding_model_key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {embedding_model_key}")

        self.model_config = EMBEDDING_MODELS[embedding_model_key]
        self.model_key = embedding_model_key

        print(f"Initializing recommendation engine with {embedding_model_key}...")

        # Load embedding model
        model_name = self.model_config["model_name"]

        # Check if it's a sentence-transformer or raw BERT model
        if model_name.startswith("sentence-transformers/"):
            self.embed_model = SentenceTransformer(model_name)
            self.is_sentence_transformer = True
        else:
            # Raw BERT-style model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(self.device)
            self.is_sentence_transformer = False

        # Connect to ChromaDB
        chroma_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", self.model_config["chroma_dir"]
        )

        if not os.path.exists(chroma_dir):
            raise FileNotFoundError(
                f"ChromaDB not found for {embedding_model_key} at {chroma_dir}\n"
                f"Please run Phase 2 with this embedding model first."
            )

        client = chromadb.PersistentClient(path=chroma_dir)
        collection_name = self.model_config["collection_name"]

        try:
            self.collection = client.get_collection(collection_name)
            print(f"Connected to ChromaDB '{collection_name}': {self.collection.count():,} talks available.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load collection '{collection_name}' from {chroma_dir}\n"
                f"Error: {e}\n"
                f"Please run Phase 2 with the {embedding_model_key} model first."
            )

    def _encode_text(self, text):
        """Encode text to embedding vector."""
        if self.is_sentence_transformer:
            return self.embed_model.encode([text])[0].tolist()
        else:
            # Manual BERT encoding with mean pooling
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Mean pooling over tokens
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).squeeze()

                # Normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                return embedding.cpu().numpy().tolist()

    def _sentiment_alignment(self, talk_meta, user_emotions):
        if not user_emotions:
            return 1.0
        scores = []
        for emotion, user_val in user_emotions.items():
            talk_val = float(talk_meta.get(emotion, 0.0))
            scores.append(1.0 - abs(user_val - talk_val))
        return sum(scores) / len(scores)

    def recommend(self, query_text, user_emotions=None, top_k=5):
        """
        query_text: natural language input from user
        user_emotions: dict like {"joy": 0.7, "sadness": 0.1, ...}
        Returns list of dicts with talk metadata and scores.
        """
        query_vec = self._encode_text(query_text)

        n_candidates = min(top_k * 6, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        for i, meta in enumerate(results["metadatas"][0]):
            # ChromaDB cosine space returns distance = 1 - cosine_similarity
            cosine_sim = max(0.0, 1.0 - results["distances"][0][i])
            sentiment_score = self._sentiment_alignment(meta, user_emotions or {})
            combined = CONTENT_WEIGHT * cosine_sim + SENTIMENT_WEIGHT * sentiment_score

            candidates.append({
                "title": meta.get("title", ""),
                "speaker": meta.get("speaker", ""),
                "url": meta.get("url", ""),
                "description": meta.get("description", ""),
                "views": meta.get("views", 0),
                "duration": meta.get("duration", 0),
                "polarity": meta.get("polarity", ""),
                "polarity_score": round(float(meta.get("polarity_score", 0.5)), 3),
                "emotions": {e: round(float(meta.get(e, 0.0)), 3) for e in EMOTION_COLS},
                "cosine_sim": round(cosine_sim, 4),
                "sentiment_alignment": round(sentiment_score, 4),
                "combined_score": round(combined, 4),
            })

        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates[:top_k]


def _fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def test_engine():
    if not os.path.exists(CHROMA_DIR):
        print(f"ERROR: ChromaDB not found at {CHROMA_DIR}")
        print("Run Phase 2 first: py -3.12 src/phase2_sentiment_embeddings.py")
        raise SystemExit(1)

    engine = RecommendationEngine()

    queries = [
        ("I want to feel inspired about climate change", {"joy": 0.6, "sadness": 0.2}),
        ("Something funny and uplifting about technology", {"joy": 0.8, "neutral": 0.2}),
        ("I feel anxious and want to understand mental health", {"fear": 0.5, "sadness": 0.4}),
        ("Curious about the universe and space exploration", {"neutral": 0.5, "joy": 0.4}),
    ]

    for query, emotions in queries:
        print(f"\n{'='*60}")
        print(f"Query:  {query}")
        print(f"Mood:   {emotions}")
        results = engine.recommend(query, emotions, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  {i}. {r['title']}")
            print(f"     Speaker: {r['speaker']} | {_fmt_duration(r['duration'])} | {r['views']:,} views")
            print(f"     Score: {r['combined_score']} (content={r['cosine_sim']}, sentiment={r['sentiment_alignment']})")
            print(f"     {r['url']}")


if __name__ == "__main__":
    test_engine()

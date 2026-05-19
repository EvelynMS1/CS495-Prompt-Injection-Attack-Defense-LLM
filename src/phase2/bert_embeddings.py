"""
Phase 2 — Step 2: Generate BERT content embeddings.

Uses bert-base-uncased with mean pooling over the last hidden state.
Input text: title + description (concise, captures semantic content).
Output: numpy array of shape (N, 768).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import BERT_MODEL, EMBED_BATCH_SIZE


class BERTEmbedder:
    def __init__(self, model_name: str = BERT_MODEL):
        print(f"Loading BERT model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"  Running on: {self.device}")

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Expand mask to match hidden state dimensions, then zero out padding
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        all_embeddings = []

        for start in tqdm(range(0, len(texts), batch_size), desc="BERT embedding"):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        print(f"Embedding matrix shape: {embeddings.shape}")
        return embeddings


def generate_embeddings(df, embedder: BERTEmbedder = None) -> np.ndarray:
    if embedder is None:
        embedder = BERTEmbedder()
    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    print(f"\nGenerating BERT embeddings for {len(texts):,} talks...")
    return embedder.embed(texts)


if __name__ == "__main__":
    import pandas as pd
    from config import CLEANED_CSV

    df = pd.read_csv(CLEANED_CSV).head(5)
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    embedder = BERTEmbedder()
    embs = generate_embeddings(df, embedder)
    print(f"Sample embedding (first 5 dims): {embs[0, :5]}")

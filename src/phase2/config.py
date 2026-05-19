"""
Phase 2 configuration — all paths and model constants in one place.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "..", "data")

# -- I/O paths ----------------------------------------------------------------
CLEANED_CSV  = os.path.join(_DATA, "cleaned_ted_talks.csv")
ENRICHED_CSV = os.path.join(_DATA, "enriched_ted_talks_bert.csv")
CHROMA_DIR   = os.path.join(_DATA, "chromadb_bert")

# -- Models -------------------------------------------------------------------
BERT_MODEL      = "bert-base-uncased"          # content embeddings (768-dim)
POLARITY_MODEL  = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL   = "SamLowe/roberta-base-go_emotions"

# -- Processing ---------------------------------------------------------------
TRANSCRIPT_CHARS = 1500   # chars of transcript fed to sentiment models
EMBED_BATCH_SIZE = 32     # rows per BERT forward pass
CHROMA_BATCH     = 100    # rows per ChromaDB insert call

# -- Emotion schema -----------------------------------------------------------
EMOTION_COLS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# GoEmotions 28 labels → 7 broad categories
GO_MAP = {
    "joy": "joy", "amusement": "joy", "excitement": "joy", "gratitude": "joy",
    "admiration": "joy", "approval": "joy", "love": "joy", "pride": "joy",
    "relief": "joy", "optimism": "joy", "caring": "joy",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
    "remorse": "sadness",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
    "neutral": "neutral", "curiosity": "neutral", "desire": "neutral",
    "embarrassment": "neutral",
}

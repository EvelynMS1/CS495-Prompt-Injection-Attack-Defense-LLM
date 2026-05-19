"""
Model Configuration for TED Talk Recommender
Defines available models for embeddings and sentiment analysis
"""

# Available embedding models
EMBEDDING_MODELS = {
    "MiniLM-L6 (Fast, 384-dim)": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "speed": "Fast",
        "collection_name": "ted_talks",
        "chroma_dir": "chromadb",
    },
    "BERT Base (Accurate, 768-dim)": {
        "model_name": "bert-base-uncased",
        "dimensions": 768,
        "speed": "Medium",
        "collection_name": "ted_talks_bert",
        "chroma_dir": "chromadb_bert",
    },
    "MPNet Base (High Quality, 768-dim)": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "speed": "Medium",
        "collection_name": "ted_talks_mpnet",
        "chroma_dir": "chromadb_mpnet",
    },
    "DistilBERT (Balanced, 768-dim)": {
        "model_name": "sentence-transformers/distilbert-base-nli-mean-tokens",
        "dimensions": 768,
        "speed": "Fast",
        "collection_name": "ted_talks_distilbert",
        "chroma_dir": "chromadb_distilbert",
    },
}

# Available sentiment analysis models
SENTIMENT_MODELS = {
    "DistilBERT SST-2 (Binary Sentiment)": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sentiment-analysis",
        "labels": ["NEGATIVE", "POSITIVE"],
    },
    "RoBERTa Twitter (Social Media)": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "task": "sentiment-analysis",
        "labels": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
    },
    "BERT Base Multilingual": {
        "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
        "task": "sentiment-analysis",
        "labels": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    },
}

# Available emotion detection models
EMOTION_MODELS = {
    "GoEmotions RoBERTa (28 emotions)": {
        "model_name": "SamLowe/roberta-base-go_emotions",
        "num_emotions": 28,
        "aggregated": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    },
    "DistilRoBERTa (7 emotions)": {
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "num_emotions": 7,
        "aggregated": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    },
}

# Default selections
DEFAULT_EMBEDDING = "MiniLM-L6 (Fast, 384-dim)"
DEFAULT_SENTIMENT = "DistilBERT SST-2 (Binary Sentiment)"
DEFAULT_EMOTION = "GoEmotions RoBERTa (28 emotions)"

# Emotion label mapping for GoEmotions -> 7 broad categories
GO_EMOTIONS_MAP = {
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

EMOTION_COLS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

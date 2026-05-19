# TED Talk Sentiment-Based Recommendation System - Complete Pipeline Flowchart

## Executive Summary

This document provides a comprehensive visual and technical flowchart of the TED Talk Sentiment-Based Recommendation System. The pipeline transforms raw TED talk data into an intelligent, emotion-aware recommendation engine accessible through an interactive chatbot interface.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE OVERVIEW                              │
└─────────────────────────────────────────────────────────────────────────────┘

   Raw Dataset                Phase 1              Phase 2                 Phase 3              Phase 4
┌──────────────┐         ┌─────────────┐      ┌──────────────┐      ┌──────────────────┐   ┌────────────┐
│              │         │             │      │              │      │                  │   │            │
│  TED Talks   │────────>│    Data     │─────>│  Sentiment   │─────>│  Recommendation  │──>│  Chatbot   │
│   Dataset    │         │  Cleaning   │      │ & Embeddings │      │     Engine       │   │     UI     │
│  (4000+)     │         │             │      │              │      │                  │   │ (Streamlit)│
│              │         │   + EDA     │      │  + Vector    │      │   Hybrid Score   │   │            │
└──────────────┘         └─────────────┘      │    Store     │      │   (70% + 30%)    │   └────────────┘
                                              └──────────────┘      └──────────────────┘
     4,000+ talks            3,900 talks         3,957 talks          Real-time Query        User Interface
   Multiple langs          English only      + 7 emotions        Semantic + Sentiment     Emotion Tracking
                          TF-IDF features    + 384/768-dim              Search
```

---

## Detailed Phase-by-Phase Flow

### Phase 1: Data Cleaning & Feature Engineering

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: DATA CLEANING                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: data/data/2020-05-01/ted_talks_en.csv
├─ Original: ~4,000+ TED talks
├─ Languages: Multiple (en, es, fr, etc.)
└─ Columns: talk_id, title, description, transcript, speaker_1, topics,
            occupations, views, duration, comments, recorded_date, published_date

                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD DATA                                                         │
│  ─────────────────                                                         │
│  pd.read_csv("data/data/2020-05-01/ted_talks_en.csv")                    │
│  Initial Shape: (4000+, 16 columns)                                       │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: EXPLORATORY DATA ANALYSIS (EDA)                                   │
│  ────────────────────────────────────────                                  │
│  ├─ Missing Values Analysis                                                │
│  ├─ Numeric Summaries (views, duration, comments)                          │
│  ├─ Visualizations Generated:                                              │
│  │  ├─ views_distribution.png (log scale histogram)                       │
│  │  ├─ duration_distribution.png                                          │
│  │  ├─ top_topics.png (horizontal bar chart, top 20)                      │
│  │  └─ speaker_occupations.png (horizontal bar chart, top 20)             │
│  └─ Output Directory: data/eda_plots/                                      │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: DATA CLEANING                                                     │
│  ──────────────────                                                        │
│  ├─ Filter: native_lang == "en"                                           │
│  ├─ Drop: rows with missing 'transcript'                                  │
│  ├─ Fill NaN: description, title, topics, occupations → ""               │
│  ├─ Parse Dates: recorded_date, published_date → datetime                │
│  ├─ Cap Views: views_capped = min(views, 99th_percentile)                │
│  ├─ Filter Duration: 60s ≤ duration ≤ 6000s                              │
│  ├─ Parse Lists: topics → topics_list (ast.literal_eval)                 │
│  └─ Parse Dicts: occupations → occupations_list (parse_dict_values)      │
│                                                                            │
│  Result: ~3,900 English talks with valid transcripts                      │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: FEATURE ENGINEERING                                               │
│  ────────────────────────                                                  │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │ TF-IDF on Titles                                        │              │
│  │ ─────────────────                                       │              │
│  │ TfidfVectorizer(max_features=200, ngram_range=(1,2))  │              │
│  │ Output: 200 columns (e.g., tfidf_title_0...199)       │              │
│  └─────────────────────────────────────────────────────────┘              │
│                          ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │ TF-IDF on Descriptions                                  │              │
│  │ ───────────────────────                                 │              │
│  │ TfidfVectorizer(max_features=300, ngram_range=(1,2))  │              │
│  │ Output: 300 columns (e.g., tfidf_desc_0...299)        │              │
│  └─────────────────────────────────────────────────────────┘              │
│                          ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │ Multi-Hot Encode Topics                                 │              │
│  │ ────────────────────────                                │              │
│  │ MultiLabelBinarizer().fit_transform(topics_list)       │              │
│  │ Output: ~47 binary columns (e.g., topic_Technology)    │              │
│  └─────────────────────────────────────────────────────────┘              │
│                          ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │ Multi-Hot Encode Occupations                            │              │
│  │ ─────────────────────────────                           │              │
│  │ MultiLabelBinarizer().fit_transform(occupations_list)  │              │
│  │ Output: Multiple binary columns (e.g., occ_Writer)     │              │
│  └─────────────────────────────────────────────────────────┘              │
│                                                                            │
│  Total New Features: 200 + 300 + 47+ topics + occupations ≈ 500+ cols    │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: SAVE CLEANED DATA                                                 │
│  ──────────────────────                                                    │
│  df.to_csv("data/cleaned_ted_talks.csv", index=False)                    │
│                                                                            │
│  Output Shape: (~3,900 rows, 500+ columns)                                │
└────────────────────────────────────────────────────────────────────────────┘

OUTPUT: data/cleaned_ted_talks.csv
```

---

### Phase 2: Sentiment Analysis & Embeddings Generation

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: SENTIMENT ANALYSIS & EMBEDDINGS                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: data/cleaned_ted_talks.csv (~3,900 English TED talks)

                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD CLEANED DATA                                                 │
│  ──────────────────────                                                    │
│  df = pd.read_csv("data/cleaned_ted_talks.csv")                          │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: SENTIMENT ANALYSIS (TWO MODELS)                                   │
│  ────────────────────────────────────                                      │
│                                                                            │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ MODEL 1: Polarity Detection                          │                 │
│  │ ────────────────────────────                         │                 │
│  │ distilbert-base-uncased-finetuned-sst-2-english     │                 │
│  │                                                      │                 │
│  │ Input: transcript[:1500]  # First 1500 chars       │                 │
│  │ Task: Binary sentiment classification               │                 │
│  │ Device: CUDA / CPU                                  │                 │
│  │                                                      │                 │
│  │ Output Columns:                                     │                 │
│  │ ├─ polarity: "POSITIVE" / "NEGATIVE"               │                 │
│  │ └─ polarity_score: float (confidence)              │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ MODEL 2: Emotion Detection                           │                 │
│  │ ────────────────────────                             │                 │
│  │ SamLowe/roberta-base-go_emotions                    │                 │
│  │                                                      │                 │
│  │ Input: transcript[:1500]  # First 1500 chars       │                 │
│  │ Task: 28-label emotion classification               │                 │
│  │ Device: CUDA / CPU                                  │                 │
│  │                                                      │                 │
│  │ Label Aggregation (28 → 7):                        │                 │
│  │ ├─ joy: [joy, amusement, excitement, gratitude,    │                 │
│  │ │        admiration, approval, love, pride,        │                 │
│  │ │        relief, optimism, caring]                 │                 │
│  │ ├─ sadness: [sadness, grief, disappointment,       │                 │
│  │ │            remorse]                              │                 │
│  │ ├─ anger: [anger, annoyance, disapproval]          │                 │
│  │ ├─ disgust: [disgust]                              │                 │
│  │ ├─ fear: [fear, nervousness]                       │                 │
│  │ ├─ surprise: [surprise, realization, confusion]    │                 │
│  │ └─ neutral: [neutral, curiosity, desire,           │                 │
│  │              embarrassment]                         │                 │
│  │                                                      │                 │
│  │ Normalization: L1 norm (sum = 1.0)                 │                 │
│  │                                                      │                 │
│  │ Output Columns (7):                                 │                 │
│  │ ├─ anger: float [0, 1]                             │                 │
│  │ ├─ disgust: float [0, 1]                           │                 │
│  │ ├─ fear: float [0, 1]                              │                 │
│  │ ├─ joy: float [0, 1]                               │                 │
│  │ ├─ neutral: float [0, 1]                           │                 │
│  │ ├─ sadness: float [0, 1]                           │                 │
│  │ └─ surprise: float [0, 1]                          │                 │
│  └──────────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: EMBEDDING GENERATION (TWO OPTIONS)                                │
│  ───────────────────────────────────────                                   │
│                                                                            │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ OPTION A (Default): MiniLM Embeddings                │                 │
│  │ ──────────────────────────────────────               │                 │
│  │ Model: sentence-transformers/all-MiniLM-L6-v2       │                 │
│  │ Dimensions: 384                                      │                 │
│  │ Batch Size: 64                                       │                 │
│  │                                                      │                 │
│  │ Input Text: title + " " + description               │                 │
│  │                                                      │                 │
│  │ Process:                                             │                 │
│  │ texts = (df["title"] + " " + df["description"])    │                 │
│  │         .tolist()                                   │                 │
│  │ embeddings = model.encode(texts, batch_size=64)     │                 │
│  │                                                      │                 │
│  │ Output Shape: (3957, 384)                           │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          OR                                                │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ OPTION B (Alternative): BERT Embeddings              │                 │
│  │ ────────────────────────────────────                 │                 │
│  │ Model: bert-base-uncased                            │                 │
│  │ Dimensions: 768                                      │                 │
│  │ Batch Size: 32                                       │                 │
│  │                                                      │                 │
│  │ Input Text: title + " " + description               │                 │
│  │                                                      │                 │
│  │ Process:                                             │                 │
│  │ texts = (df["title"] + " " + df["description"])    │                 │
│  │         .tolist()                                   │                 │
│  │ tokenizer → input_ids, attention_mask               │                 │
│  │ model(input_ids, attention_mask)                    │                 │
│  │ → last_hidden_state                                 │                 │
│  │ → mean pooling with attention mask                  │                 │
│  │ → L2 normalize                                      │                 │
│  │                                                      │                 │
│  │ Output Shape: (3957, 768)                           │                 │
│  └──────────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: CHROMADB VECTOR STORE CREATION                                    │
│  ────────────────────────────────────────                                  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Initialize ChromaDB Client                           │                 │
│  │ ───────────────────────────                          │                 │
│  │ Storage Path (Option A): data/chromadb/             │                 │
│  │ Storage Path (Option B): data/chromadb_bert/        │                 │
│  │                                                      │                 │
│  │ client = chromadb.PersistentClient(path=CHROMA_DIR) │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Create Collection                                     │                 │
│  │ ──────────────────                                    │                 │
│  │ Collection Name (Option A): "ted_talks"             │                 │
│  │ Collection Name (Option B): "ted_talks_bert"        │                 │
│  │                                                      │                 │
│  │ Configuration:                                       │                 │
│  │ ├─ Distance Metric: cosine                          │                 │
│  │ ├─ Index Type: HNSW (Hierarchical Navigable        │                 │
│  │ │               Small World)                        │                 │
│  │ └─ Embedding Dimension: 384 or 768                  │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Prepare Metadata (Per Entry)                         │                 │
│  │ ─────────────────────────                            │                 │
│  │ {                                                    │                 │
│  │   "talk_id": str,                                   │                 │
│  │   "title": str[:500],  # Truncated                  │                 │
│  │   "speaker": str[:200],                             │                 │
│  │   "views": int,                                     │                 │
│  │   "duration": int,                                  │                 │
│  │   "url": str[:500],                                 │                 │
│  │   "description": str[:1000],                        │                 │
│  │   "polarity": "POSITIVE" / "NEGATIVE" / "NEUTRAL",  │                 │
│  │   "polarity_score": float,                          │                 │
│  │   "anger": float,                                   │                 │
│  │   "disgust": float,                                 │                 │
│  │   "fear": float,                                    │                 │
│  │   "joy": float,                                     │                 │
│  │   "neutral": float,                                 │                 │
│  │   "sadness": float,                                 │                 │
│  │   "surprise": float                                 │                 │
│  │ }                                                    │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Batch Insert to ChromaDB                             │                 │
│  │ ─────────────────────────                            │                 │
│  │ Batch Size: 100 entries                             │                 │
│  │                                                      │                 │
│  │ for start in range(0, 3957, 100):                   │                 │
│  │     batch = data[start:start+100]                   │                 │
│  │     collection.add(                                 │                 │
│  │         ids=batch_ids,                              │                 │
│  │         documents=batch_texts,                      │                 │
│  │         metadatas=batch_metadata,                   │                 │
│  │         embeddings=batch_embeddings                 │                 │
│  │     )                                               │                 │
│  │                                                      │                 │
│  │ Total Entries: 3,957 TED talks                      │                 │
│  └──────────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: SAVE ENRICHED DATA                                                │
│  ────────────────────────                                                  │
│  df.to_csv("data/enriched_ted_talks.csv", index=False)                   │
│         OR                                                                 │
│  df.to_csv("data/enriched_ted_talks_bert.csv", index=False)              │
│                                                                            │
│  New Columns Added (9):                                                    │
│  ├─ polarity, polarity_score                                             │
│  └─ anger, disgust, fear, joy, neutral, sadness, surprise                │
└────────────────────────────────────────────────────────────────────────────┘

OUTPUTS:
├─ data/enriched_ted_talks.csv (or enriched_ted_talks_bert.csv)
└─ data/chromadb/ (or chromadb_bert/)
   └─ Collection: 3,957 entries with 384-dim or 768-dim embeddings
```

---

### Phase 3: Recommendation Engine

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3: RECOMMENDATION ENGINE                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

INPUT: data/chromadb/ (3,957 vector entries)

                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  INITIALIZATION (One-Time Setup)                                           │
│  ────────────────────────────────                                          │
│  engine = RecommendationEngine()                                           │
│                                                                            │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Load Embedding Model                                 │                 │
│  │ ─────────────────────                                │                 │
│  │ sentence-transformers/all-MiniLM-L6-v2              │                 │
│  │ Dimensions: 384                                      │                 │
│  │ Device: CUDA / CPU                                   │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Connect to ChromaDB                                  │                 │
│  │ ────────────────────                                 │                 │
│  │ Path: data/chromadb/                                │                 │
│  │ Collection: "ted_talks"                             │                 │
│  │ Count: 3,957 entries                                │                 │
│  └──────────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  RECOMMENDATION FLOW (Per Query)                                           │
│  ────────────────────────────────                                          │
│                                                                            │
│  INPUT PARAMETERS:                                                         │
│  ├─ query: str (user's search query)                                     │
│  ├─ user_emotions: dict[str, float] (7 emotion scores)                   │
│  └─ top_k: int (default=5, number of results to return)                  │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: QUERY EMBEDDING                                                   │
│  ────────────────────────                                                  │
│  query_vector = embed_model.encode([query])[0]                            │
│  Shape: (384,)                                                             │
│  Dtype: float32                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: VECTOR SEARCH (HNSW ANN)                                          │
│  ──────────────────────────────────                                        │
│  results = collection.query(                                               │
│      query_embeddings=[query_vector],                                     │
│      n_results=top_k * 6,  # Over-fetch for re-ranking                   │
│      include=["documents", "metadatas", "distances"]                      │
│  )                                                                         │
│                                                                            │
│  Example: top_k=5 → retrieve 30 candidates                                │
│                                                                            │
│  ChromaDB Returns:                                                         │
│  ├─ distances: List[float] (cosine distances, lower = more similar)      │
│  ├─ documents: List[str] (title + " " + description)                     │
│  └─ metadatas: List[dict] (all stored metadata)                          │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: HYBRID RE-RANKING                                                 │
│  ──────────────────────                                                    │
│  For each candidate talk:                                                  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ 3a. Content Similarity Score                         │                 │
│  │ ─────────────────────────────                        │                 │
│  │ cosine_sim = 1.0 - chroma_distance                  │                 │
│  │                                                      │                 │
│  │ Example:                                             │                 │
│  │ distance = 0.2 → cosine_sim = 0.8                   │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ 3b. Sentiment Alignment Score                        │                 │
│  │ ──────────────────────────────                       │                 │
│  │ If user_emotions provided:                           │                 │
│  │                                                      │                 │
│  │   For each of 7 emotions:                           │                 │
│  │     diff = |user_emotion - talk_emotion|            │                 │
│  │     alignment = 1.0 - diff                          │                 │
│  │                                                      │                 │
│  │   sentiment_score = mean(7 alignment scores)        │                 │
│  │                                                      │                 │
│  │ Else (no user emotions):                            │                 │
│  │   sentiment_score = 1.0  # Neutral/no penalty       │                 │
│  │                                                      │                 │
│  │ Example:                                             │                 │
│  │ User: {joy: 0.7, sadness: 0.1, ...}                │                 │
│  │ Talk: {joy: 0.8, sadness: 0.05, ...}               │                 │
│  │ → sentiment_score ≈ 0.92                            │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ 3c. Combined Score (Weighted Hybrid)                 │                 │
│  │ ─────────────────────────────────                    │                 │
│  │ CONTENT_WEIGHT = 0.7                                │                 │
│  │ SENTIMENT_WEIGHT = 0.3                              │                 │
│  │                                                      │                 │
│  │ combined_score = (CONTENT_WEIGHT * cosine_sim)      │                 │
│  │                + (SENTIMENT_WEIGHT * sentiment_score)│                 │
│  │                                                      │                 │
│  │ Example:                                             │                 │
│  │ 0.7 * 0.8 + 0.3 * 0.92 = 0.56 + 0.276 = 0.836      │                 │
│  └──────────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: SORT & RETURN TOP-K                                               │
│  ─────────────────────────────                                             │
│  candidates.sort(key=lambda x: x["combined_score"], reverse=True)         │
│  return candidates[:top_k]                                                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: List[dict]                                                        │
│  ───────────────────                                                       │
│  [                                                                         │
│    {                                                                       │
│      "title": str,                                                        │
│      "speaker": str,                                                      │
│      "url": str,                                                          │
│      "description": str,                                                  │
│      "views": int,                                                        │
│      "duration": int,                                                     │
│      "polarity": "POSITIVE" / "NEGATIVE" / "NEUTRAL",                    │
│      "polarity_score": float,                                            │
│      "emotions": {                                                        │
│        "anger": float, "disgust": float, "fear": float,                 │
│        "joy": float, "neutral": float, "sadness": float,                │
│        "surprise": float                                                 │
│      },                                                                   │
│      "cosine_sim": float,           # Content relevance                  │
│      "sentiment_alignment": float,   # Emotion match                     │
│      "combined_score": float        # Weighted hybrid score              │
│    },                                                                      │
│    ... (top_k results)                                                    │
│  ]                                                                         │
└────────────────────────────────────────────────────────────────────────────┘

OUTPUT: Ranked list of top-k TED talk recommendations
```

---

### Phase 4: Interactive Chatbot

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             PHASE 4: STREAMLIT CHATBOT                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

INITIALIZATION (Cached Resources)

┌────────────────────────────────────────────────────────────────────────────┐
│  @st.cache_resource: LOAD MODELS & ENGINE                                  │
│  ──────────────────────────────────────────                                │
│                                                                            │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Load User Emotion Model                              │                 │
│  │ ────────────────────────                             │                 │
│  │ j-hartmann/emotion-english-distilroberta-base       │                 │
│  │                                                      │                 │
│  │ Task: 7-class emotion classification                │                 │
│  │ Input: User query (max 256 tokens)                  │                 │
│  │ Output: 7 emotion scores (normalized)               │                 │
│  │ Device: CUDA / CPU                                   │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                          ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ Load RecommendationEngine                            │                 │
│  │ ──────────────────────────                           │                 │
│  │ from phase3_recommendation import                   │                 │
│  │     RecommendationEngine                            │                 │
│  │                                                      │                 │
│  │ engine = RecommendationEngine()                     │                 │
│  │ ├─ MiniLM model loaded                              │                 │
│  │ └─ ChromaDB connection established                  │                 │
│  └──────────────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  SESSION STATE INITIALIZATION                                              │
│  ─────────────────────────────                                             │
│  st.session_state.emotion_context = {                                     │
│      "anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0,             │
│      "neutral": 1.0, "sadness": 0.0, "surprise": 0.0                     │
│  }                                                                         │
│  st.session_state.messages = []  # Chat history                           │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  USER INTERACTION LOOP                                                     │
│  ──────────────────────                                                    │
│                                                                            │
│  User types query → st.chat_input("Ask for TED Talk recommendations...")  │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: ANALYZE USER EMOTION (Real-Time)                                  │
│  ──────────────────────────────────────                                    │
│  inputs = emotion_tokenizer(                                               │
│      user_query,                                                           │
│      return_tensors="pt",                                                  │
│      truncation=True,                                                      │
│      max_length=256                                                        │
│  )                                                                         │
│                                                                            │
│  outputs = emotion_model(**inputs)                                         │
│  logits = outputs.logits                                                   │
│  probs = torch.nn.functional.softmax(logits, dim=1)[0]                    │
│                                                                            │
│  current_emotions = {                                                      │
│      "anger": float(probs[0]),                                            │
│      "disgust": float(probs[1]),                                          │
│      "fear": float(probs[2]),                                             │
│      "joy": float(probs[3]),                                              │
│      "neutral": float(probs[4]),                                          │
│      "sadness": float(probs[5]),                                          │
│      "surprise": float(probs[6])                                          │
│  }                                                                         │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: UPDATE EMOTION CONTEXT (EMA Smoothing)                            │
│  ────────────────────────────────────────────                              │
│  ALPHA = 0.4  # Exponential moving average smoothing factor               │
│                                                                            │
│  For each emotion:                                                         │
│    new_value = ALPHA * current_emotion                                    │
│              + (1 - ALPHA) * previous_context_emotion                     │
│                                                                            │
│  Example:                                                                  │
│  ├─ Previous context: {joy: 0.3}                                          │
│  ├─ Current query: {joy: 0.8}                                             │
│  └─ Updated: 0.4*0.8 + 0.6*0.3 = 0.32 + 0.18 = 0.50                      │
│                                                                            │
│  Purpose: Smooths emotion transitions across conversation turns           │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: GET RECOMMENDATIONS                                               │
│  ────────────────────────────                                              │
│  recommendations = engine.recommend(                                       │
│      query=user_query,                                                    │
│      user_emotions=emotion_context,  # Smoothed emotions                 │
│      top_k=5                                                              │
│  )                                                                         │
│                                                                            │
│  (Invokes Phase 3 pipeline: embed → search → re-rank)                    │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: FORMAT RESPONSE                                                   │
│  ────────────────────────                                                  │
│  For each recommendation:                                                  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │ **{rank}. {title}** by {speaker}                     │                 │
│  │ {description}                                        │                 │
│  │                                                      │                 │
│  │ 📊 Views: {views:,} | ⏱ Duration: {MM}:{SS}        │                 │
│  │ 💡 Polarity: {polarity} ({polarity_score:.2f})     │                 │
│  │ 🎭 Top Emotions: {top_3_emotions}                   │                 │
│  │ 🔗 [Watch on TED]({url})                            │                 │
│  │                                                      │                 │
│  │ 🎯 Content Match: {cosine_sim:.1%}                  │                 │
│  │ ❤️ Emotion Alignment: {sentiment_alignment:.1%}     │                 │
│  │ ⭐ Combined Score: {combined_score:.1%}             │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                                                                            │
│  Markdown formatting applied for rich display                             │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: DISPLAY IN CHAT                                                   │
│  ────────────────────────                                                  │
│  st.chat_message("assistant").markdown(response)                          │
│                                                                            │
│  Chat history updated:                                                     │
│  st.session_state.messages.append({                                       │
│      "role": "user", "content": user_query                                │
│  })                                                                        │
│  st.session_state.messages.append({                                       │
│      "role": "assistant", "content": response                             │
│  })                                                                        │
└────────────────────────────────────────────────────────────────────────────┘
                                    ▼

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: UPDATE SIDEBAR (Live Emotion Tracker)                             │
│  ───────────────────────────────────────────────                           │
│  For each of 7 emotions:                                                   │
│    st.sidebar.progress(                                                    │
│        value=emotion_context[emotion],                                    │
│        text=f"{emotion.capitalize()}: {value:.1%}"                        │
│    )                                                                       │
│                                                                            │
│  Visual bars show real-time emotional state                               │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  USER INTERFACE LAYOUT                                                     │
│  ──────────────────────                                                    │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │                    STREAMLIT APP                            │          │
│  ├─────────────────────────────────────────────────────────────┤          │
│  │  Title: "TED Talk Sentiment-Based Recommender Chatbot"    │          │
│  │                                                             │          │
│  │  Sidebar:                                                   │          │
│  │  ├─ "Your Current Emotional Context"                       │          │
│  │  ├─ [Progress Bar] Joy: 45%                               │          │
│  │  ├─ [Progress Bar] Neutral: 30%                           │          │
│  │  ├─ [Progress Bar] Surprise: 15%                          │          │
│  │  └─ ... (7 emotion bars total)                            │          │
│  │                                                             │          │
│  │  Main Area:                                                 │          │
│  │  ┌───────────────────────────────────────────────┐         │          │
│  │  │ [User] "I want to feel inspired"              │         │          │
│  │  ├───────────────────────────────────────────────┤         │          │
│  │  │ [Assistant] Here are 5 recommendations:       │         │          │
│  │  │                                               │         │          │
│  │  │ **1. Do schools kill creativity?**            │         │          │
│  │  │ by Ken Robinson                               │         │          │
│  │  │ ...                                           │         │          │
│  │  └───────────────────────────────────────────────┘         │          │
│  │                                                             │          │
│  │  [Chat Input Box] "Ask for TED Talk recommendations..."    │          │
│  └─────────────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────────────┘

OUTPUT: Interactive chatbot with emotion-aware recommendations
```

---

## Complete End-to-End Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE PIPELINE DATA FLOW                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘

DATA STAGE                  STORAGE LOCATION                    SHAPE/SIZE
──────────────────────────────────────────────────────────────────────────────────────

1. RAW DATASET
   ted_talks_en.csv    ───> data/data/2020-05-01/              (4000+, 16 cols)
   ├─ All languages          ted_talks_en.csv                   Multiple langs
   └─ Mixed quality                                             Some missing data

                              ▼ PHASE 1: CLEANING ▼

2. CLEANED DATASET
   cleaned_ted_talks   ───> data/                               (~3900, 500+ cols)
   ├─ English only          cleaned_ted_talks.csv               English only
   ├─ Valid transcripts                                         No missing text
   ├─ TF-IDF features                                          200 title features
   └─ Multi-hot topics                                         300 desc features

                              ▼ PHASE 2: SENTIMENT ▼

3. SENTIMENT SCORES
   Polarity            ───> Columns in enriched CSV             (~3900, 509 cols)
   ├─ polarity                                                  POSITIVE/NEGATIVE
   ├─ polarity_score                                           [0.0, 1.0]
   └─ 7 emotions                                               anger, disgust,
                                                                fear, joy, neutral,
                                                                sadness, surprise

                              ▼ PHASE 2: EMBEDDINGS ▼

4. EMBEDDINGS
   Vector embeddings   ───> data/chromadb/                      (3957, 384)
   ├─ MiniLM (384-dim)      collection: "ted_talks"            or (3957, 768)
   └─ BERT (768-dim)        collection: "ted_talks_bert"       for BERT option

                              ▼ PHASE 2: VECTOR STORE ▼

5. VECTOR DATABASE
   ChromaDB            ───> data/chromadb/                      3,957 entries
   ├─ Persistent            or data/chromadb_bert/              HNSW index
   ├─ HNSW indexed                                             Cosine distance
   └─ Rich metadata                                            17 fields/entry

                              ▼ PHASE 3: RECOMMENDATION ▼

6. QUERY RESULTS
   Ranked list         ───> In-memory (returned by engine)      (top_k, 20 fields)
   ├─ Top-K talks                                               Default K=5
   ├─ Hybrid scores                                            Combined score
   └─ Full metadata                                            URL, emotions, etc.

                              ▼ PHASE 4: CHATBOT UI ▼

7. USER INTERFACE
   Streamlit app       ───> Browser (localhost:8501)            Interactive
   ├─ Chat history          Session state                       Persistent
   ├─ Emotion context       EMA smoothed                       7 emotions
   └─ Live recommendations  Rendered markdown                   Clickable links


KEY METRICS AT EACH STAGE:
─────────────────────────────
Phase 1:  4000+ → 3900 talks (filtering to English, valid transcripts)
Phase 2:  3900 → 3957 talks (sentiment + embeddings)
Phase 3:  3957 → top_k results (typically 5-10 recommendations)
Phase 4:  Real-time query processing (~100-500ms per query)
```

---

## Models & Technologies Summary

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          MODELS & TECHNOLOGIES USED                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘

PHASE     MODEL/TECHNOLOGY                         PURPOSE                    DIMS/OUTPUT
─────────────────────────────────────────────────────────────────────────────────────────

Phase 1   • scikit-learn TfidfVectorizer          Title feature extraction   200 features
          • scikit-learn TfidfVectorizer          Desc feature extraction    300 features
          • scikit-learn MultiLabelBinarizer      Topic encoding             ~47 features
          • Pandas, NumPy                         Data manipulation          N/A

Phase 2   • distilbert-base-uncased-finetuned-   Binary polarity            2 classes
            sst-2-english
          • SamLowe/roberta-base-go_emotions      28-label → 7 emotions      7 scores
          • sentence-transformers/               Semantic embeddings         384-dim
            all-MiniLM-L6-v2 (DEFAULT)
          • bert-base-uncased (ALTERNATIVE)       Semantic embeddings         768-dim
          • ChromaDB                              Vector database             HNSW, cosine

Phase 3   • sentence-transformers/               Query encoding              384-dim
            all-MiniLM-L6-v2
          • ChromaDB                              Vector search               ANN search
          • Custom Hybrid Ranker                  Score combination           0.7 + 0.3

Phase 4   • j-hartmann/emotion-english-          Real-time user emotions     7 classes
            distilroberta-base
          • Streamlit                             Web UI framework            Web app
          • Custom EMA Tracker                    Emotion smoothing           α = 0.4
```

---

## Performance Characteristics

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            PERFORMANCE METRICS                                        │
└──────────────────────────────────────────────────────────────────────────────────────┘

PHASE         OPERATION                    ESTIMATED TIME           COMPLEXITY
────────────────────────────────────────────────────────────────────────────────────────

Phase 1       Data cleaning                2-5 minutes              O(n)
              TF-IDF computation           1-2 minutes              O(n * vocab_size)
              Total Phase 1                3-7 minutes              One-time

Phase 2       Sentiment analysis           10-20 minutes (CPU)      O(n * seq_len)
              (DistilBERT + RoBERTa)       3-5 minutes (GPU)
              Embedding generation         5-10 minutes (CPU)       O(n * seq_len)
              (MiniLM or BERT)             1-2 minutes (GPU)
              ChromaDB insertion           30-60 seconds            O(n log n)
              Total Phase 2                15-30 min (CPU)          One-time
                                          5-8 min (GPU)

Phase 3       Query embedding              10-50 ms                 O(seq_len)
              Vector search (HNSW)         20-100 ms                O(log n)
              Hybrid re-ranking            5-20 ms                  O(k)
              Total per query              35-170 ms                Real-time

Phase 4       User emotion analysis        10-30 ms                 O(seq_len)
              Recommendation call          35-170 ms                (Phase 3)
              UI rendering                 5-15 ms                  O(k)
              Total per interaction        50-215 ms                Real-time

DATASET SIZE: 3,957 TED talks
EMBEDDING DIMS: 384 (MiniLM) or 768 (BERT)
STORAGE SIZE: ~50-100 MB (ChromaDB + enriched CSV)
```

---

## Execution Commands Reference

```bash
# Full Pipeline (All Phases)
py -3.12 src/run_pipeline.py

# Individual Phases
py -3.12 src/phase1_data_cleaning.py
py -3.12 src/phase2_sentiment_embeddings.py      # MiniLM version
py -3.12 src/phase2/run_phase2.py                # BERT version
py -3.12 src/phase3_recommendation.py            # Smoke test only
py -3.12 -m streamlit run src/phase4_chatbot.py  # Launch chatbot

# Modular Phase 2 Demo (BERT)
py -3.12 src/phase2/demo.py                      # Interactive CLI demo
```

---

## Key Design Decisions & Rationale

### 1. Hybrid Ranking (70% Content + 30% Sentiment)
**Why**: Pure semantic search misses emotional alignment; pure sentiment misses content relevance. The 70/30 split balances both, giving content slight priority while still accounting for emotional context.

### 2. Emotion Context Tracking (EMA with α=0.4)
**Why**: Raw per-message emotions are noisy; EMA smoothing with α=0.4 creates continuity across conversation turns while still being responsive to new input.

### 3. Dual Embedding Options (MiniLM vs BERT)
**Why**: MiniLM is 2x faster and smaller (384-dim) but BERT offers higher expressiveness (768-dim). The modular design lets users choose based on speed vs quality needs.

### 4. Over-fetching Strategy (top_k × 6)
**Why**: ChromaDB returns results by pure cosine distance, but hybrid re-ranking changes order. Retrieving 6× ensures high-quality candidates survive re-ranking.

### 5. Truncation to 1,500 Characters
**Why**: DistilBERT/RoBERTa have 512-token limits. 1,500 chars (~375 words) balances covering intro/conclusion of transcripts with model constraints.

### 6. Smart Caching (Check Before Recompute)
**Why**: Phase 2 sentiment analysis is expensive. Checking for existing enriched CSV with required columns prevents redundant computation.

### 7. Batch Processing (32-64 embeddings, 100 ChromaDB inserts)
**Why**: Balances memory usage with GPU/API efficiency. Larger batches risk OOM errors; smaller batches underutilize hardware.

---

## Data Privacy & Security Notes

- **No Personal Data**: TED talks dataset is public domain
- **No User Tracking**: Emotion analysis runs locally, no data sent externally
- **Local Storage**: All embeddings and databases stored in local `/data` directory
- **No API Keys Required**: All models loaded from Hugging Face Hub (public)

---

## Future Enhancement Opportunities

1. **User Feedback Loop**: Collect thumbs up/down to fine-tune ranking weights
2. **Multi-Query Reranking**: Allow users to chain queries (e.g., "inspiring but not sad")
3. **Speaker Diversity**: Add diversity penalty to avoid recommending same speaker
4. **Topic Filters**: Let users filter by TED talk topics (Technology, Science, etc.)
5. **Transcript Search**: Add keyword search within transcripts (currently title+desc only)
6. **Export History**: Allow users to export chat history and recommendations
7. **A/B Testing**: Compare MiniLM vs BERT effectiveness with user studies

---

## Troubleshooting Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: ted_talks_en.csv` | Raw dataset not downloaded | Download from Kaggle TED Ultimate Dataset |
| `CUDA out of memory` | GPU insufficient for batch size | Reduce batch size or use CPU |
| `ChromaDB collection not found` | Phase 2 not run | Execute `py -3.12 src/phase2_sentiment_embeddings.py` |
| `Streamlit ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |
| Slow recommendations | CPU-only inference | Enable GPU with `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

---

## Architecture Diagram (ASCII)

```
                    ┌─────────────────────────────────────┐
                    │     USER (Browser/Streamlit UI)     │
                    └──────────────┬──────────────────────┘
                                   │
                                   │ User query
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       PHASE 4: CHATBOT (UI)         │
                    │  ┌─────────────────────────────┐    │
                    │  │ Emotion Model (DistilRoBERTa)│   │
                    │  │ → Analyze user emotion      │    │
                    │  │ → Update EMA context        │    │
                    │  └─────────────────────────────┘    │
                    └──────────────┬──────────────────────┘
                                   │
                                   │ Query + Emotions
                                   ▼
                    ┌─────────────────────────────────────┐
                    │   PHASE 3: RECOMMENDATION ENGINE    │
                    │  ┌─────────────────────────────┐    │
                    │  │ MiniLM Embedding Model      │    │
                    │  │ → Embed query (384-dim)     │    │
                    │  └─────────────────────────────┘    │
                    │               │                      │
                    │               ▼                      │
                    │  ┌─────────────────────────────┐    │
                    │  │ ChromaDB Vector Search      │    │
                    │  │ → HNSW ANN (cosine)         │    │
                    │  │ → Retrieve top_k * 6        │    │
                    │  └─────────────────────────────┘    │
                    │               │                      │
                    │               ▼                      │
                    │  ┌─────────────────────────────┐    │
                    │  │ Hybrid Re-Ranker            │    │
                    │  │ → 0.7 * cosine_sim          │    │
                    │  │ → 0.3 * sentiment_align     │    │
                    │  │ → Sort & return top_k       │    │
                    │  └─────────────────────────────┘    │
                    └──────────────┬──────────────────────┘
                                   │
                                   │ Ranked recommendations
                                   ▼
                    ┌─────────────────────────────────────┐
                    │       DATA STORAGE LAYER            │
                    │  ┌─────────────────────────────┐    │
                    │  │ ChromaDB (Vector DB)        │    │
                    │  │ • 3,957 embeddings          │    │
                    │  │ • 384/768 dimensions        │    │
                    │  │ • HNSW indexed              │    │
                    │  │ • Rich metadata (17 fields) │    │
                    │  └─────────────────────────────┘    │
                    │               ▲                      │
                    │               │                      │
                    │  Populated by Phase 2                │
                    └───────────────────────────────────────┘

                              ▲
                              │ One-time preprocessing
                              │
                    ┌─────────────────────────────────────┐
                    │    PHASE 2: SENTIMENT & EMBEDDINGS  │
                    │  ┌─────────────────────────────┐    │
                    │  │ DistilBERT (Polarity)       │    │
                    │  │ → POSITIVE/NEGATIVE         │    │
                    │  └─────────────────────────────┘    │
                    │  ┌─────────────────────────────┐    │
                    │  │ RoBERTa GoEmotions          │    │
                    │  │ → 7 emotions (normalized)   │    │
                    │  └─────────────────────────────┘    │
                    │  ┌─────────────────────────────┐    │
                    │  │ MiniLM / BERT               │    │
                    │  │ → Generate embeddings       │    │
                    │  └─────────────────────────────┘    │
                    └──────────────┬──────────────────────┘
                                   ▲
                                   │
                                   │ Cleaned dataset
                    ┌──────────────┴──────────────────────┐
                    │    PHASE 1: DATA CLEANING           │
                    │  ┌─────────────────────────────┐    │
                    │  │ Filter to English           │    │
                    │  │ Clean missing values        │    │
                    │  │ TF-IDF feature extraction   │    │
                    │  │ Multi-hot topic encoding    │    │
                    │  └─────────────────────────────┘    │
                    └──────────────┬──────────────────────┘
                                   ▲
                                   │
                                   │ Raw dataset
                    ┌──────────────┴──────────────────────┐
                    │  TED TALKS DATASET (Kaggle)         │
                    │  • 4000+ talks                      │
                    │  • Multiple languages               │
                    │  • Mixed quality                    │
                    └─────────────────────────────────────┘
```

---

## Summary

This TED Talk Sentiment-Based Recommendation System is a **complete, production-ready pipeline** that:

✅ **Cleans and preprocesses** 4,000+ TED talks down to 3,957 high-quality English talks
✅ **Analyzes sentiment** using state-of-the-art transformer models (DistilBERT, RoBERTa)
✅ **Generates embeddings** with MiniLM (fast, 384-dim) or BERT (accurate, 768-dim)
✅ **Stores vectors efficiently** in ChromaDB with HNSW indexing for fast retrieval
✅ **Ranks intelligently** using hybrid scoring (70% content + 30% emotion alignment)
✅ **Delivers recommendations** through an interactive Streamlit chatbot with real-time emotion tracking

**Total pipeline runtime**: 15-30 minutes (one-time setup), then **<200ms per query** (real-time)

**Key innovation**: Emotion-aware recommendations that consider both what users are searching for (semantic content) and how they're feeling (emotional tone).

---

*Document Generated: 2026-05-18*
*Project: CS495-Prompt-Injection-Attack-Defense-LLM (TED Talk Recommender)*
*Version: 1.0*

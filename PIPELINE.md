# Pipeline Documentation
## TED Talk Sentiment-Based Recommendation System

---

## Execution Order

```
run_pipeline.py  (orchestrator)
       │
       ▼
phase1_data_cleaning.py
       │  data/cleaned_ted_talks.csv
       ▼
phase2_sentiment_embeddings.py
       │  data/enriched_ted_talks.csv
       │  data/chromadb/  (vector DB)
       ▼
phase3_recommendation.py
       │  RecommendationEngine class (imported)
       ▼
phase4_chatbot.py  (Streamlit UI, live)
```

---

## `run_pipeline.py` — Orchestrator

**Purpose:** Runs all four phases in sequence as subprocesses. Halts on any failure.
**Input:** Nothing — just invokes the scripts in order.
**Output:** Delegates to each phase; launches Streamlit at the end via `streamlit run`.
**How to start the whole system:**

```bash
py -3.12 src/run_pipeline.py
```

---

## Phase 1 — `phase1_data_cleaning.py`

**Input:** `data/data/2020-05-01/ted_talks_en.csv` (raw TED Talk dataset)

**Internal pipeline (5 steps):**

| Step | Function | What it does |
|---|---|---|
| 1 | `load_data()` | Reads raw CSV → DataFrame (~4k+ rows) |
| 2 | `run_eda()` | Computes missing-value stats, generates 3 plots (distributions, top topics, top occupations) → saves to `data/eda_plots/` |
| 3 | `clean()` | Filters to English-only, drops rows without transcripts, fills missing text fields, parses dates, caps views at 99th percentile, drops talks under 60s or over 6000s |
| 4 | `build_features()` | TF-IDF on titles (200 features) + descriptions (300 features); multi-hot encodes topics and speaker occupations; concatenates all into the DataFrame |
| 5 | `save()` | Writes `data/cleaned_ted_talks.csv` |

**Output:** `data/cleaned_ted_talks.csv` — cleaned DataFrame with topic/occupation feature columns appended.

**Dependency:** None. This is the pipeline entry point.

---

## Phase 2 — `phase2_sentiment_embeddings.py`

**Input:** `data/cleaned_ted_talks.csv` (Phase 1 output)
**Depends on Phase 1 completing first** — exits with error if the CSV is missing.

**Internal pipeline (3 steps):**

| Step | Function | Model used | Output |
|---|---|---|---|
| 2a | `enrich_with_sentiment()` | DistilBERT (polarity) + RoBERTa GoEmotions (7 emotions) | Adds `polarity`, `polarity_score`, and 7 emotion columns (joy, anger, sadness, fear, disgust, surprise, neutral) |
| 2b | `generate_embeddings()` | `all-MiniLM-L6-v2` | 384-dim vector per talk from `title + description` |
| 2c | `load_chromadb()` | ChromaDB | Inserts all talks (embeddings + metadata) into the `ted_talks` collection |

**Outputs:**
- `data/enriched_ted_talks.csv` — CSV with all sentiment columns added
- `data/chromadb/` — persistent vector database with 3,957 talks

**Smart skip:** If `enriched_ted_talks.csv` already exists with sentiment columns, it skips the expensive model inference and goes straight to embedding + ChromaDB load.

---

## Phase 3 — `phase3_recommendation.py`

**Input:** `data/chromadb/` (Phase 2 output — must exist)
**Depends on Phase 2.**

**What it provides:**

- `RecommendationEngine` class — the core retrieval logic, imported by Phase 4
- `engine.recommend(query_text, user_emotions, top_k)` encodes the query with `all-MiniLM-L6-v2`, fetches candidates from ChromaDB, then ranks by a **hybrid score**:

```
combined_score = 0.7 × cosine_similarity + 0.3 × sentiment_alignment
```

When run standalone, `test_engine()` fires 4 sample queries as a smoke test.

**Output:** Returns ranked list of talk dicts (title, speaker, URL, scores). No files written.

---

## Phase 4 — `phase4_chatbot.py`

**Input:** Live user text from Streamlit chat input
**Imports directly from:** `phase3_recommendation.RecommendationEngine`
**Depends on Phase 3 (and therefore Phase 2's ChromaDB).**

**Conversation loop:**

```
User types message
       │
       ▼
analyze_emotions()          ← j-hartmann DistilRoBERTa (7-class, real-time)
       │
       ▼
update_emotion_context()    ← exponential moving average (α=0.4) across turns
       │
       ▼
engine.recommend()          ← hybrid retrieval from ChromaDB
       │
       ▼
build_bot_response()        ← formats markdown with titles, links, scores
       │
       ▼
Streamlit renders results + sidebar emotion bar chart
```

**Sentiment context tracking:** Each new message blends with the running emotional profile (40% current, 60% history), so the recommender adapts if the user's mood shifts mid-conversation.

---

## End-to-End Data Flow Summary

```
ted_talks_en.csv
    → [Phase 1] clean + feature engineer
    → cleaned_ted_talks.csv
    → [Phase 2] DistilBERT polarity + RoBERTa emotions + MiniLM embeddings
    → enriched_ted_talks.csv + ChromaDB (ted_talks collection)
    → [Phase 3] RecommendationEngine (cosine + sentiment hybrid)
    → [Phase 4] Streamlit chatbot with live emotion tracking
```

The only file you must run from scratch (unless the CSVs and ChromaDB already exist) is:

```bash
py -3.12 src/run_pipeline.py
```

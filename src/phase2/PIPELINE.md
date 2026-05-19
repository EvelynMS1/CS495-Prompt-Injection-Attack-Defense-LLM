# Phase 2 Pipeline
## Sentiment Analysis + BERT Embeddings + ChromaDB

---

## How to Run

```bash
# From the project root — run the full Phase 2 pipeline
py -3.12 src/phase2/run_phase2.py
```

Phase 1 must have already produced `data/cleaned_ted_talks.csv` before this runs.

---

## File Map

```
src/phase2/
├── config.py              — all paths and model constants
├── data_loader.py         — Step 1: load & validate Phase 1 CSV
├── sentiment_analysis.py  — Step 2: DistilBERT polarity + RoBERTa GoEmotions
├── bert_embeddings.py     — Step 3: BERT content embeddings (768-dim)
├── vector_store.py        — Step 4: ChromaDB ingestion
├── run_phase2.py          — orchestrator (entry point)
└── PIPELINE.md            — this file
```

---

## Execution Order & Data Flow

```
data/cleaned_ted_talks.csv          ← Phase 1 output (required input)
         │
         ▼
[Step 1] data_loader.py
         load()
         • Reads CSV, validates required columns
         • Fills NaN in title / description / transcript
         │
         ▼  DataFrame (~3,900 rows)
         │
[Step 2] sentiment_analysis.py
         enrich_with_sentiment()
         │
         ├─ DistilBERT  (distilbert-base-uncased-finetuned-sst-2-english)
         │   _polarity()  → polarity (POSITIVE/NEGATIVE), polarity_score
         │
         └─ RoBERTa GoEmotions  (SamLowe/roberta-base-go_emotions)
             _emotions() → 28 raw labels collapsed to 7 emotions:
                           anger, disgust, fear, joy,
                           neutral, sadness, surprise
         │
         ▼  DataFrame + polarity + 7 emotion columns
         │
         └──► data/enriched_ted_talks_bert.csv  (saved to disk)
         │
[Step 3] bert_embeddings.py
         BERTEmbedder.embed()
         • Model: bert-base-uncased (768-dim output)
         • Input text: title + description per talk
         • Tokenized, padded, truncated at 512 tokens
         • Forward pass through BERT
         • Mean pool last hidden state (ignoring padding tokens)
         • Processed in batches of 32
         │
         ▼  numpy array  (N × 768)
         │
[Step 4] vector_store.py
         load_into_chromadb()
         • Creates / recreates collection: ted_talks_bert
         • Cosine similarity space (hnsw:space = cosine)
         • Stores per talk:
             – BERT embedding (768-dim vector)
             – document: "title description"
             – metadata: talk_id, title, speaker, views, duration,
                         url, description, polarity, polarity_score,
                         anger, disgust, fear, joy, neutral,
                         sadness, surprise
         • Inserted in batches of 100
         │
         ▼
data/chromadb_bert/                 ← persistent vector database
  collection: ted_talks_bert (~3,900 entries)
```

---

## Inputs & Outputs per Module

| Module | Input | Output |
|---|---|---|
| `data_loader.py` | `data/cleaned_ted_talks.csv` | Validated DataFrame |
| `sentiment_analysis.py` | DataFrame | DataFrame + `polarity`, `polarity_score`, 7 emotion columns |
| `bert_embeddings.py` | DataFrame (`title`, `description`) | `numpy` array (N × 768) |
| `vector_store.py` | DataFrame + embeddings array | `data/chromadb_bert/` collection |
| `run_phase2.py` | — (orchestrates all above) | enriched CSV + ChromaDB |

---

## Smart Skip

If `data/enriched_ted_talks_bert.csv` already exists and contains both `polarity` and all 7 emotion columns, `run_phase2.py` skips the sentiment inference step (which is slow on CPU — ~hours for 3,900 talks) and loads the cached CSV directly before proceeding to embedding + ChromaDB.

---

## Key Design Decisions vs Phase 1

| Aspect | Phase 1 approach | Phase 2 approach |
|---|---|---|
| Content representation | TF-IDF (200 title + 300 description features) | BERT mean-pooled embeddings (768-dim) |
| Semantic understanding | Bag-of-words, sparse | Contextual, dense |
| Vocabulary | Fixed vocab from training data | Shared `bert-base-uncased` vocabulary |
| Similarity search | Not applicable (features only) | Cosine similarity in ChromaDB |
| Emotion scoring | Not applicable | DistilBERT + RoBERTa GoEmotions |

---

## ChromaDB Collection Schema

| Field | Type | Source |
|---|---|---|
| `id` | string | `talk_id` from CSV |
| `embedding` | float[768] | BERT mean pool of title + description |
| `document` | string | `title + description` |
| `title` | string | metadata |
| `speaker` | string | metadata |
| `views` | int | metadata |
| `duration` | int | metadata (seconds) |
| `url` | string | metadata |
| `description` | string | metadata |
| `polarity` | string | DistilBERT (POSITIVE/NEGATIVE/NEUTRAL) |
| `polarity_score` | float | DistilBERT confidence |
| `joy` | float | RoBERTa GoEmotions (0–1 normalized) |
| `sadness` | float | RoBERTa GoEmotions |
| `anger` | float | RoBERTa GoEmotions |
| `fear` | float | RoBERTa GoEmotions |
| `disgust` | float | RoBERTa GoEmotions |
| `surprise` | float | RoBERTa GoEmotions |
| `neutral` | float | RoBERTa GoEmotions |

---

## Dependencies

All required packages are in the project `requirements.txt`:

```
torch
transformers
sentence-transformers
chromadb
pandas
numpy
tqdm
```

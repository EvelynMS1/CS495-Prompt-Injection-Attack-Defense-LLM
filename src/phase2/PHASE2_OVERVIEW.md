# Phase 2 — Sentiment Analysis, BERT Embeddings & ChromaDB

## Purpose

Phase 2 takes the cleaned TED Talk data from Phase 1 and enriches it with AI-generated features: sentiment/emotion scores and BERT semantic embeddings. The end result is a ChromaDB vector database that can be searched by both meaning and mood.

---

## How to Run

### Full Pipeline
```
py -3.12 src/phase2/run_phase2.py
```
Runs all 4 steps in order. If `data/enriched_ted_talks_bert.csv` already exists with sentiment columns, the expensive sentiment step is skipped automatically.

### Demo / Search
```
py -3.12 src/phase2/demo.py                    # runs 4 preset sample queries
py -3.12 src/phase2/demo.py "your query here"  # runs a single custom query
```
Requires the ChromaDB collection to exist (i.e., the full pipeline has been run at least once).

### Individual Modules
Each module can also be run standalone for testing:
```
py -3.12 src/phase2/data_loader.py
py -3.12 src/phase2/bert_embeddings.py
py -3.12 src/phase2/sentiment_analysis.py
py -3.12 src/phase2/vector_store.py
```

---

## Prerequisites

- Phase 1 must have been run first: `py -3.12 src/phase1_data_cleaning.py`
- Input file: `data/cleaned_ted_talks.csv`

---

## Files

| File | Role |
|---|---|
| `config.py` | All file paths, model names, and constants in one place |
| `data_loader.py` | Step 1 — loads and validates the Phase 1 CSV |
| `sentiment_analysis.py` | Step 2 — runs DistilBERT (polarity) and RoBERTa GoEmotions (7 emotions) on each transcript |
| `bert_embeddings.py` | Step 3 — generates 768-dim BERT embeddings from title + description |
| `vector_store.py` | Step 4 — inserts embeddings and metadata into ChromaDB |
| `run_phase2.py` | Orchestrator — runs Steps 1–4 in order |
| `demo.py` | Search tool — queries ChromaDB using BERT + emotion alignment scoring |

### File Details

**`config.py`**
Central configuration. Defines input/output paths, the 3 AI models, batch sizes, and the mapping from GoEmotions' 28 fine-grained labels down to 7 broad emotion categories (anger, disgust, fear, joy, neutral, sadness, surprise).

**`data_loader.py`**
Loads `cleaned_ted_talks.csv` and validates that required columns (`title`, `description`, `transcript`, `views`, `duration`) are present. Exits with a clear error if Phase 1 has not been run.

**`sentiment_analysis.py`**
Runs two NLP models on the first 1,500 characters of each transcript:
- **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) → `polarity` (POSITIVE/NEGATIVE) + `polarity_score`
- **RoBERTa GoEmotions** (`SamLowe/roberta-base-go_emotions`) → scores for 7 emotion categories per talk

**`bert_embeddings.py`**
Generates a 768-dimensional semantic embedding per talk using `bert-base-uncased`. Input text is `title + description`. Uses mean pooling over BERT's last hidden layer, processed in batches of 32.

**`vector_store.py`**
Inserts all data into ChromaDB (collection: `ted_talks_bert`, cosine similarity space). Each entry stores:
- 768-dim BERT vector for semantic search
- Metadata: title, speaker, views, duration, URL, description, polarity, and all 7 emotion scores

**`run_phase2.py`**
Orchestrates the full pipeline (Steps 1–4). Skips sentiment analysis if the enriched CSV already exists and contains all expected columns.

**`demo.py`**
Standalone search tool. Encodes a query with BERT, retrieves candidates from ChromaDB, then re-ranks using a weighted score: **70% semantic similarity + 30% emotion alignment**. Supports preset sample queries or a custom query passed as a command-line argument.

---

## Data Flow

```
data/cleaned_ted_talks.csv       (Phase 1 output)
         |
   data_loader.py                loads & validates
         |
   sentiment_analysis.py         adds polarity + 7 emotion columns
         |
data/enriched_ted_talks_bert.csv
         |
   bert_embeddings.py            generates 768-dim vectors
         |
   vector_store.py               writes to ChromaDB
         |
data/chromadb_bert/              (Phase 2 output — used by Phase 3+)
```

---

## Outputs

| File/Folder | Description |
|---|---|
| `data/enriched_ted_talks_bert.csv` | Cleaned data + polarity + 7 emotion scores |
| `data/chromadb_bert/` | Persistent ChromaDB collection with BERT vectors and metadata |

---

## Models Used

| Model | Purpose |
|---|---|
| `bert-base-uncased` | 768-dim content embeddings |
| `distilbert-base-uncased-finetuned-sst-2-english` | POSITIVE/NEGATIVE polarity |
| `SamLowe/roberta-base-go_emotions` | 28-label emotions collapsed to 7 categories |

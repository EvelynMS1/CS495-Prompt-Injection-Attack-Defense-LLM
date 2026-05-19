# Phase 2 — Sentiment Analysis, Embeddings & ChromaDB
## Execution Order and Data Flow

Phase 2 has two implementations that share the same sentiment models but differ
in embedding strategy and output collection. Both are documented here.

---

## Implementation A — `src/phase2_sentiment_embeddings.py`
### (MiniLM 384-dim, collection: `ted_talks`)

**How to run:**
```powershell
py -3.12 src/phase2_sentiment_embeddings.py
```

**Input:** `data/cleaned_ted_talks.csv`
**Outputs:** `data/enriched_ted_talks.csv` + `data/chromadb/`

### Models

| Model | Task | Text input | Output |
|---|---|---|---|
| `distilbert-base-uncased-finetuned-sst-2-english` | Polarity classification | First 1,500 chars of transcript | `POSITIVE`/`NEGATIVE` + confidence score |
| `SamLowe/roberta-base-go_emotions` | Emotion classification | First 1,500 chars of transcript | 28 raw labels → 7 normalized scores |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic embedding | `title + " " + description` | 384-dim float vector |

### Execution Order

```
main()
  │
  ├── Smart skip check
  │     If data/enriched_ted_talks.csv exists AND has "polarity" + "joy" columns:
  │       Load cached CSV → skip Steps 2a entirely
  │     Else:
  │       Load data/cleaned_ted_talks.csv → run Steps 2a
  │
  ├── Step 2a — enrich_with_sentiment(df, sentiment_pipe, emotion_pipe)
  │              → data/enriched_ted_talks.csv
  │
  ├── Step 2b — generate_embeddings(df, embed_model)
  │              → numpy array (N, 384)
  │
  └── Step 2c — load_chromadb(df, embeddings)
                 → data/chromadb/  collection: ted_talks
```

### Step 2a — `enrich_with_sentiment()`

Iterates every row in the DataFrame. Both models receive only the first **1,500 characters**
of the transcript — fast on CPU, captures the opening tone of each talk.

```
For each row:

  transcript[:1500]
        │
        ├──► _get_polarity(sentiment_pipe, text)
        │       DistilBERT sentiment-analysis pipeline
        │       result["label"] → "POSITIVE" or "NEGATIVE"
        │       Score normalization:
        │         POSITIVE → result["score"] as-is
        │         NEGATIVE → 1.0 - result["score"]
        │                    (converts to positive-polarity scale, rounded to 4 dp)
        │       Returns: (label: str, score: float)
        │       Appended to: polarities[], polarity_scores[]
        │
        └──► _get_emotions(emotion_pipe, text)
               RoBERTa GoEmotions pipeline (top_k=None → all 28 labels returned)
               Raw output: [{"label": "amusement", "score": 0.31}, …]
               │
               ▼ Collapse 28 labels → 7 broad categories via _GO_MAP:
                   joy      ← joy, amusement, excitement, gratitude, admiration,
                               approval, love, pride, relief, optimism, caring
                   sadness  ← sadness, grief, disappointment, remorse
                   anger    ← anger, annoyance, disapproval
                   disgust  ← disgust
                   fear     ← fear, nervousness
                   surprise ← surprise, realization, confusion
                   neutral  ← neutral, curiosity, desire, embarrassment
               │
               ▼ Aggregate: sum scores of all labels that map to each category
               ▼ Normalize: each category score ÷ total (all 7 sum to ≈ 1.0)
               Returns: {"joy": 0.52, "anger": 0.08, …}
               Appended to: emotion_records[]

After all rows complete:
  df["polarity"]        = polarities
  df["polarity_score"]  = polarity_scores
  emotion_df = pd.DataFrame(emotion_records, index=df.index).fillna(0.0)
  df = pd.concat([df, emotion_df], axis=1)
  df.to_csv("data/enriched_ted_talks.csv", index=False)
```

**New columns added to DataFrame:**
`polarity`, `polarity_score`, `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`

---

### Step 2b — `generate_embeddings(df, embed_model)`

Only the title and description are embedded — the transcript is not used here.

```
df["title"].fillna("") + " " + df["description"].fillna("")
        │
        ▼  List of N strings (one per talk)
        │
        ▼  SentenceTransformer.encode(
               texts,
               batch_size=64,
               show_progress_bar=True
           )
           Tokenizes internally
           Runs MiniLM forward pass in batches of 64
           Mean-pools sentence embeddings
        │
        ▼  numpy array  shape: (N, 384)
```

---

### Step 2c — `load_chromadb(df, embeddings)`

```
chromadb.PersistentClient(path="data/chromadb/")
        │
        ▼  Delete existing "ted_talks" collection if present
        ▼  Create collection "ted_talks"
             metadata: {"hnsw:space": "cosine"}   ← cosine similarity index
        │
        ▼  For each row, assemble:
             id        = str(talk_id)
             document  = title + " " + description
             metadata  = {
                 talk_id, title (≤500 chars), speaker (≤200 chars),
                 views (int), duration (int, seconds),
                 url (≤500 chars), description (≤1000 chars),
                 polarity, polarity_score,
                 anger, disgust, fear, joy, neutral, sadness, surprise
             }
             embedding = embeddings[i].tolist()   ← 384 floats
        │
        ▼  collection.add() in batches of 100
        │
        data/chromadb/
          Collection: "ted_talks"
          ~3,957 entries
          Each entry: 384-dim vector + metadata dict
```

---
---

## Implementation B — `src/phase2/run_phase2.py`
### (BERT 768-dim, collection: `ted_talks_bert`)

A modular refactor of Phase 2 that uses full `bert-base-uncased` embeddings (768-dim)
instead of MiniLM (384-dim). Writes to a separate ChromaDB collection and CSV so both
implementations can coexist.

**How to run:**
```powershell
py -3.12 src/phase2/run_phase2.py
```

**Input:** `data/cleaned_ted_talks.csv`
**Outputs:** `data/enriched_ted_talks_bert.csv` + `data/chromadb_bert/`

### Module Map

```
src/phase2/
├── config.py              — all path and model constants
├── data_loader.py         — Step 1: load & validate the Phase 1 CSV
├── sentiment_analysis.py  — Step 2: DistilBERT polarity + RoBERTa GoEmotions
├── bert_embeddings.py     — Step 3: BERT content embeddings (768-dim)
├── vector_store.py        — Step 4: ChromaDB ingestion
└── run_phase2.py          — orchestrator (entry point)
```

### Constants (`config.py`)

| Constant | Value |
|---|---|
| `BERT_MODEL` | `bert-base-uncased` |
| `POLARITY_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` |
| `EMOTION_MODEL` | `SamLowe/roberta-base-go_emotions` |
| `TRANSCRIPT_CHARS` | `1500` |
| `EMBED_BATCH_SIZE` | `32` |
| `CHROMA_BATCH` | `100` |

### Execution Order

```
run_phase2.main()
  │
  ├── Step 1 — data_loader.load()
  │              pd.read_csv(CLEANED_CSV)
  │              Fills NaN in title, description, transcript
  │              Validates required columns are present
  │              → validated DataFrame
  │
  ├── Step 2 — sentiment_analysis.enrich_with_sentiment(df)
  │              Smart skip: if enriched_ted_talks_bert.csv already exists
  │              with "polarity" and all 7 emotion columns → load cached CSV
  │              Otherwise runs DistilBERT + RoBERTa GoEmotions (same logic as
  │              Implementation A, same models, same 1,500-char truncation)
  │              → DataFrame + polarity + 7 emotion columns
  │              → saved to data/enriched_ted_talks_bert.csv
  │
  ├── Step 3 — bert_embeddings.generate_embeddings(df, embedder)
  │              → numpy array (N, 768)
  │
  └── Step 4 — vector_store.load_into_chromadb(df, embeddings)
                 → data/chromadb_bert/  collection: ted_talks_bert
```

### Step 3 — `bert_embeddings.BERTEmbedder.embed()`

This step differs from Implementation A. Instead of SentenceTransformer, a raw
`bert-base-uncased` model is loaded and mean pooling is applied manually.

```
BERTEmbedder.__init__()
  AutoTokenizer.from_pretrained("bert-base-uncased")
  AutoModel.from_pretrained("bert-base-uncased")
  device = cuda if available, else cpu
  model.eval()

generate_embeddings(df, embedder)
  texts = df["title"].fillna("") + " " + df["description"].fillna("")
  BERTEmbedder.embed(texts, batch_size=32)
    │
    ▼  For each batch of 32 texts:
         tokenizer(
           batch,
           padding=True,
           truncation=True,
           max_length=512,
           return_tensors="pt"
         )
         → input_ids, attention_mask tensors moved to device
         │
         ▼  with torch.no_grad():
              model(**encoded)
              → output.last_hidden_state  shape: (batch, seq_len, 768)
         │
         ▼  _mean_pool(last_hidden_state, attention_mask)
              attention_mask expanded to (batch, seq_len, 768)
              Padding positions zeroed out
              Sum across seq_len dimension → (batch, 768)
              Divide by count of non-padding tokens per row
              → (batch, 768) mean-pooled vectors
         │
         ▼  pooled.cpu().numpy() → appended to all_embeddings[]

    np.vstack(all_embeddings)
    → numpy array  shape: (N, 768)
```

### Step 4 — `vector_store.load_into_chromadb()`

Same structure as Implementation A but targets a different path and collection name.

```
chromadb.PersistentClient(path="data/chromadb_bert/")
  Delete + recreate collection "ted_talks_bert"
  metadata: {"hnsw:space": "cosine"}

  Per row → { id, document, metadata, 768-dim embedding }
  Inserted in batches of 100

  data/chromadb_bert/
    Collection: "ted_talks_bert"
    ~3,900 entries
    Each entry: 768-dim BERT vector + metadata dict
```

---

## Implementation Comparison

| Aspect | Implementation A | Implementation B |
|---|---|---|
| Entry point | `phase2_sentiment_embeddings.py` | `src/phase2/run_phase2.py` |
| Embedding model | `all-MiniLM-L6-v2` | `bert-base-uncased` |
| Embedding dimension | 384 | 768 |
| Pooling method | SentenceTransformer internal | Manual mean pool (last hidden state) |
| Batch size (embed) | 64 | 32 |
| Sentiment models | DistilBERT + RoBERTa GoEmotions | DistilBERT + RoBERTa GoEmotions |
| Enriched CSV | `enriched_ted_talks.csv` | `enriched_ted_talks_bert.csv` |
| ChromaDB path | `data/chromadb/` | `data/chromadb_bert/` |
| Collection name | `ted_talks` | `ted_talks_bert` |
| Consumed by | Phase 3 & 4 (live chatbot) | Standalone / research use |

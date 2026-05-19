# Phase 2 Modular Pipeline

## Sentiment Analysis + BERT Embeddings + ChromaDB

### `src/phase2/`

---

## How to Run

```powershell
py -3.12 src/phase2/run_phase2.py
```

Phase 1 must have already produced `data/cleaned_ted_talks.csv` before this runs.

---

## Module Map

```
src/phase2/
├── config.py              — all path and model constants
├── data_loader.py         — Step 1: load & validate Phase 1 CSV
├── sentiment_analysis.py  — Step 2: DistilBERT polarity + RoBERTa GoEmotions
├── bert_embeddings.py     — Step 3: BERT content embeddings (768-dim, mean pooled)
├── vector_store.py        — Step 4: ChromaDB ingestion
└── run_phase2.py          — orchestrator (entry point)
```

---

## Configuration — `config.py`

All constants are defined here and imported by every other module.

| Constant           | Value                                                     | Purpose                                          |
| ------------------ | --------------------------------------------------------- | ------------------------------------------------ |
| `CLEANED_CSV`      | `data/cleaned_ted_talks.csv`                              | Phase 1 output (input to this pipeline)          |
| `ENRICHED_CSV`     | `data/enriched_ted_talks_bert.csv`                        | Output of sentiment step                         |
| `CHROMA_DIR`       | `data/chromadb_bert/`                                     | ChromaDB persistence directory                   |
| `BERT_MODEL`       | `bert-base-uncased`                                       | Content embedding model                          |
| `POLARITY_MODEL`   | `distilbert-base-uncased-finetuned-sst-2-english`         | Polarity classifier                              |
| `EMOTION_MODEL`    | `SamLowe/roberta-base-go_emotions`                        | 28-label emotion classifier                      |
| `TRANSCRIPT_CHARS` | `1500`                                                    | Characters of transcript fed to sentiment models |
| `EMBED_BATCH_SIZE` | `32`                                                      | Rows per BERT forward pass                       |
| `CHROMA_BATCH`     | `100`                                                     | Rows per ChromaDB insert call                    |
| `EMOTION_COLS`     | `[anger, disgust, fear, joy, neutral, sadness, surprise]` | 7 target emotion categories                      |

GoEmotions label collapse map (`GO_MAP`) — 28 fine-grained labels → 7 categories:

```
joy      ← joy, amusement, excitement, gratitude, admiration,
            approval, love, pride, relief, optimism, caring
sadness  ← sadness, grief, disappointment, remorse
anger    ← anger, annoyance, disapproval
disgust  ← disgust
fear     ← fear, nervousness
surprise ← surprise, realization, confusion
neutral  ← neutral, curiosity, desire, embarrassment
```

---

## Execution Order — `run_phase2.main()`

```
run_phase2.main()
  │
  ├── Step 1 — data_loader.load()
  │
  ├── Step 2 — sentiment_analysis.enrich_with_sentiment(df)
  │     Smart skip: if ENRICHED_CSV exists with "polarity" + all 7 emotion cols
  │       → load cached CSV, skip inference
  │     Else → run full sentiment inference, save ENRICHED_CSV
  │
  ├── Step 3 — bert_embeddings.BERTEmbedder() + generate_embeddings(df, embedder)
  │
  └── Step 4 — vector_store.load_into_chromadb(df, embeddings)
```

---

## Step-by-Step Data Flow

---

### Step 1 — `data_loader.load()`

```
data/cleaned_ted_talks.csv   (Phase 1 output)
        │
        ▼  pd.read_csv(CLEANED_CSV)
        ▼  Validate required columns are present
        ▼  df["title"].fillna("")
           df["description"].fillna("")
           df["transcript"].fillna("")
        DataFrame  (~3,900 rows, all Phase 1 columns intact)
        │
        └──► passed to Step 2
```

---

### Step 2 — `sentiment_analysis.enrich_with_sentiment(df)`

Both models receive only the **first 1,500 characters** of each transcript.

```
enrich_with_sentiment(df)
  │
  ├── load_polarity_pipe()
  │     pipeline("sentiment-analysis",
  │              model=POLARITY_MODEL,
  │              truncation=True, max_length=512)
  │
  ├── load_emotion_pipe()
  │     pipeline("text-classification",
  │              model=EMOTION_MODEL,
  │              top_k=None,
  │              truncation=True, max_length=512)
  │
  └── For each row in df:
        text = str(row["transcript"])
        │
        ├──► _polarity(polarity_pipe, text)
        │      pipe(text[:1500])[0]
        │      label = "POSITIVE" or "NEGATIVE"
        │      score:
        │        POSITIVE → result["score"] as-is
        │        NEGATIVE → round(1.0 - result["score"], 4)
        │      → (label, score)
        │      appended to polarities[], polarity_scores[]
        │
        └──► _emotions(emotion_pipe, text)
               pipe(text[:1500])[0]
               → list of 28 dicts: [{"label": "amusement", "score": 0.31}, …]
               │
               ▼  For each label:
                    broad_category = GO_MAP.get(label, "neutral")
                    aggregated[broad_category] += score
               │
               ▼  Normalize:
                    total = sum(aggregated.values()) or 1.0
                    each value ÷ total   →  7 scores summing to ≈ 1.0
               → {"joy": 0.52, "anger": 0.08, …}
               appended to emotion_records[]

  After all rows:
    df = df.copy()
    df["polarity"]       = polarities
    df["polarity_score"] = polarity_scores
    emotion_df = pd.DataFrame(emotion_records, index=df.index).fillna(0.0)
    df = pd.concat([df, emotion_df], axis=1)

  Printed to stdout:
    polarity value_counts()
    mean score per emotion column

  df.to_csv(ENRICHED_CSV, index=False)
  → data/enriched_ted_talks_bert.csv

New columns added: polarity, polarity_score,
                   anger, disgust, fear, joy, neutral, sadness, surprise
```

---

### Step 3 — `bert_embeddings.generate_embeddings(df, embedder)`

Uses a raw `bert-base-uncased` model with manual mean pooling.
Only title + description are embedded — the transcript is not used here.

```
BERTEmbedder.__init__(model_name="bert-base-uncased")
  │
  ▼  AutoTokenizer.from_pretrained("bert-base-uncased")
  ▼  AutoModel.from_pretrained("bert-base-uncased")
  ▼  device = cuda if torch.cuda.is_available() else cpu
  ▼  model.to(device).eval()

generate_embeddings(df, embedder)
  │
  ▼  texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
     N strings, one per talk
  │
  ▼  embedder.embed(texts, batch_size=32)
       │
       ▼  For each batch of 32 texts:
            tokenizer(
              batch,
              padding=True,
              truncation=True,
              max_length=512,
              return_tensors="pt"
            )
            → input_ids tensor       shape: (batch_size, seq_len)
            → attention_mask tensor  shape: (batch_size, seq_len)
            Both moved to device
            │
            ▼  with torch.no_grad():
                 model(**encoded)
                 → output.last_hidden_state  shape: (batch_size, seq_len, 768)
            │
            ▼  _mean_pool(last_hidden_state, attention_mask)
                 mask = attention_mask.unsqueeze(-1)
                        .expand(last_hidden_state.size()).float()
                 Zeroes out padding token positions
                 summed = (last_hidden_state × mask).sum(dim=1)  shape: (batch, 768)
                 counts = mask.sum(dim=1).clamp(min=1e-9)        shape: (batch, 768)
                 pooled = summed / counts                         shape: (batch, 768)
            │
            ▼  pooled.cpu().numpy()
               appended to all_embeddings[]

       np.vstack(all_embeddings)
       → numpy array  shape: (N, 768)
```

---

### Step 4 — `vector_store.load_into_chromadb(df, embeddings)`

```
chromadb.PersistentClient(path=CHROMA_DIR)   → data/chromadb_bert/
  │
  ▼  Delete existing "ted_talks_bert" collection if present
  ▼  Create collection "ted_talks_bert"
       metadata: {"hnsw:space": "cosine"}    ← cosine similarity index
  │
  ▼  For each row i in df:
       talk_id = str(int(row["talk_id"])) if not NaN else str(i)

       id       = talk_id
       document = title + " " + description
       metadata = {
           talk_id       : str
           title         : str  (capped at 500 chars)
           speaker       : str  (capped at 200 chars, from speaker_1 column)
           views         : int  (0 if NaN)
           duration      : int  (seconds, 0 if NaN)
           url           : str  (capped at 500 chars)
           description   : str  (capped at 1000 chars)
           polarity      : str  (POSITIVE / NEGATIVE / NEUTRAL)
           polarity_score: float
           anger         : float
           disgust       : float
           fear          : float
           joy           : float
           neutral       : float
           sadness       : float
           surprise      : float
       }
       embedding = embeddings[i].tolist()    ← 768 floats
  │
  ▼  collection.add(ids, documents, metadatas, embeddings)
     in batches of CHROMA_BATCH (100 rows per call)
  │
  data/chromadb_bert/
    Collection: "ted_talks_bert"
    ~3,900 entries
    Each entry: 768-dim BERT vector + metadata dict
    Similarity space: cosine (HNSW index)
```

---

## Inputs and Outputs per Module

| Module                  | Input                              | Output                                                                                          |
| ----------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| `data_loader.py`        | `data/cleaned_ted_talks.csv`       | Validated DataFrame                                                                             |
| `sentiment_analysis.py` | DataFrame                          | DataFrame + `polarity`, `polarity_score`, 7 emotion columns; `data/enriched_ted_talks_bert.csv` |
| `bert_embeddings.py`    | DataFrame (`title`, `description`) | numpy array (N × 768)                                                                           |
| `vector_store.py`       | DataFrame + (N × 768) embeddings   | `data/chromadb_bert/` collection `ted_talks_bert`                                               |
| `run_phase2.py`         | — (orchestrates all above)         | Enriched CSV + ChromaDB                                                                         |

---

## ChromaDB Collection Schema

| Field            | Type       | Source                                     |
| ---------------- | ---------- | ------------------------------------------ |
| `id`             | string     | `talk_id` from CSV                         |
| `embedding`      | float[768] | BERT mean pool of `title + description`    |
| `document`       | string     | `title + description`                      |
| `title`          | string     | metadata                                   |
| `speaker`        | string     | metadata (`speaker_1` column)              |
| `views`          | int        | metadata                                   |
| `duration`       | int        | metadata (seconds)                         |
| `url`            | string     | metadata                                   |
| `description`    | string     | metadata                                   |
| `polarity`       | string     | DistilBERT (POSITIVE / NEGATIVE / NEUTRAL) |
| `polarity_score` | float      | DistilBERT confidence                      |
| `joy`            | float      | RoBERTa GoEmotions (normalized 0–1)        |
| `sadness`        | float      | RoBERTa GoEmotions                         |
| `anger`          | float      | RoBERTa GoEmotions                         |
| `fear`           | float      | RoBERTa GoEmotions                         |
| `disgust`        | float      | RoBERTa GoEmotions                         |
| `surprise`       | float      | RoBERTa GoEmotions                         |
| `neutral`        | float      | RoBERTa GoEmotions                         |

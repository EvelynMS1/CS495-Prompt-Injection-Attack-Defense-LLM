# Full Pipeline — All Phases
## TED Talk Sentiment-Based Recommendation System
### Execution Order and Data Flow

---

## Top-Level Orchestration — `src/run_pipeline.py`

```powershell
py -3.12 src/run_pipeline.py
```

`run_pipeline.py` calls `subprocess.run()` for each phase in sequence.
If any phase exits with a non-zero code, the pipeline halts immediately.

```
run_pipeline.main()
  │
  ├── subprocess → phase1_data_cleaning.py
  │       output: data/cleaned_ted_talks.csv
  │
  ├── subprocess → phase2_sentiment_embeddings.py
  │       output: data/enriched_ted_talks.csv
  │               data/chromadb/  (collection: ted_talks)
  │
  ├── subprocess → phase3_recommendation.py
  │       (smoke test only — no files written)
  │
  └── subprocess → streamlit run phase4_chatbot.py
          (live server — runs until Ctrl+C)
```

---

---

# Phase 1 — Data Collection & Cleaning
### `src/phase1_data_cleaning.py`

**Input:** `data/data/2020-05-01/ted_talks_en.csv`
**Output:** `data/cleaned_ted_talks.csv` + `data/eda_plots/*.png`
**Upstream dependency:** None. Pipeline entry point.

## Execution Order

```
main()
  ├── 1. load_data()
  ├── 2. run_eda()
  ├── 3. clean()
  ├── 4. build_features()
  └── 5. save()
```

## Data Flow

### Step 1 — `load_data()`
```
data/data/2020-05-01/ted_talks_en.csv
        │
        ▼  pd.read_csv()
        DataFrame
          All rows (English + non-English)
          Columns: talk_id, title, speaker_1, description, transcript,
                   topics, occupations, views, duration, comments,
                   native_lang, recorded_date, published_date, url
```

### Step 2 — `run_eda()` *(DataFrame unchanged — side-effects only)*
```
DataFrame (read-only)
        │
        ├── missing-value % per column          → printed to stdout
        ├── numeric summary (views/duration/comments) → printed to stdout
        ├── views + duration histograms          → data/eda_plots/distributions.png
        ├── top-20 topics bar chart              → data/eda_plots/top_topics.png
        └── top-20 occupations bar chart         → data/eda_plots/top_occupations.png
```

### Step 3 — `clean()`
```
All rows, all languages
        │
        ▼  filter native_lang == "en"
        ▼  dropna(subset=["transcript"])
        ▼  fill missing text fields ("description","title","topics","occupations") with ""
        ▼  parse recorded_date, published_date → datetime (coerce errors to NaT)
        ▼  views_capped = views.clip(upper=views.quantile(0.99))
        ▼  filter duration: 60 s ≤ duration ≤ 6000 s
        ▼  _safe_parse_list("topics")       → df["topics_list"]
        ▼  _parse_dict_values("occupations") → df["occupations_list"]
        ▼  reset_index(drop=True)
        ~3,900 rows  (English, transcript present, valid duration)
```

### Step 4 — `build_features()`
```
Cleaned DataFrame
        │
        ├── TfidfVectorizer(max_features=200, ngram_range=(1,2)) on "title"
        │     → title_df  (N × 200)  columns: tfidf_title_<term>
        │
        ├── TfidfVectorizer(max_features=300, ngram_range=(1,2)) on "description"
        │     → desc_df   (N × 300)  columns: tfidf_desc_<term>
        │
        ├── MultiLabelBinarizer on "topics_list"
        │     → topics_df (N × unique topics)  columns: topic_<label>
        │
        ├── MultiLabelBinarizer on "occupations_list"
        │     → occ_df    (N × unique occupations)  columns: occ_<label>
        │
        └── pd.concat([df, topics_df, occ_df], axis=1)
              (title_df and desc_df returned to main but not written to CSV)
```

### Step 5 — `save()`
```
Enriched DataFrame
        ▼  drop columns: topics_list, occupations_list
        ▼  df.to_csv("data/cleaned_ted_talks.csv", index=False)
        data/cleaned_ted_talks.csv
```

---

---

# Phase 2 — Sentiment Analysis, Embeddings & ChromaDB
### `src/phase2_sentiment_embeddings.py`

**Input:** `data/cleaned_ted_talks.csv`
**Output:** `data/enriched_ted_talks.csv` + `data/chromadb/` (collection: `ted_talks`)
**Upstream dependency:** Phase 1

## Models

| Model | Task | Input | Output |
|---|---|---|---|
| `distilbert-base-uncased-finetuned-sst-2-english` | Polarity | transcript[:1500 chars] | POSITIVE/NEGATIVE + score |
| `SamLowe/roberta-base-go_emotions` | Emotion (28 → 7 labels) | transcript[:1500 chars] | 7 normalized emotion scores |
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding | title + description | 384-dim float vector |

## Execution Order

```
main()
  │
  ├── smart skip: if enriched CSV exists with polarity + emotion cols → skip Step 2a
  │
  ├── Step 2a — enrich_with_sentiment()    → data/enriched_ted_talks.csv
  ├── Step 2b — generate_embeddings()      → (N, 384) numpy array
  └── Step 2c — load_chromadb()            → data/chromadb/
```

## Data Flow

### Step 2a — `enrich_with_sentiment()`
```
For each row:
  transcript[:1500]
        │
        ├──► DistilBERT
        │      _get_polarity()
        │      label = POSITIVE or NEGATIVE
        │      score: POSITIVE → raw score
        │             NEGATIVE → 1.0 - raw score
        │      → polarities[], polarity_scores[]
        │
        └──► RoBERTa GoEmotions (top_k=None → all 28 labels)
               _get_emotions()
               28 labels aggregated into 7 categories:
                 joy      ← joy, amusement, excitement, gratitude, admiration,
                             approval, love, pride, relief, optimism, caring
                 sadness  ← sadness, grief, disappointment, remorse
                 anger    ← anger, annoyance, disapproval
                 disgust  ← disgust
                 fear     ← fear, nervousness
                 surprise ← surprise, realization, confusion
                 neutral  ← neutral, curiosity, desire, embarrassment
               Aggregated by summing scores per category
               Normalized: each ÷ total  (7 scores sum to ≈ 1.0)
               → emotion_records[]

After all rows:
  df["polarity"], df["polarity_score"] appended
  emotion_df = pd.DataFrame(emotion_records).fillna(0.0)
  df = pd.concat([df, emotion_df], axis=1)
  → data/enriched_ted_talks.csv
```

### Step 2b — `generate_embeddings()`
```
df["title"] + " " + df["description"]   (one string per talk)
        │
        ▼  SentenceTransformer.encode(batch_size=64)
        (N, 384) numpy array
```

### Step 2c — `load_chromadb()`
```
PersistentClient("data/chromadb/")
  drop + recreate collection "ted_talks"  (hnsw:space = cosine)
  Per row:
    id        = str(talk_id)
    document  = title + " " + description
    metadata  = { talk_id, title, speaker, views, duration, url,
                  description, polarity, polarity_score,
                  anger, disgust, fear, joy, neutral, sadness, surprise }
    embedding = 384-dim float list
  Insert in batches of 100
  → data/chromadb/  (~3,957 entries)
```

---

---

# Phase 3 — Recommendation Engine
### `src/phase3_recommendation.py`

**Input:** `data/chromadb/` (Phase 2 output, must exist)
**Output:** Ranked list of talk dicts returned in memory. No files written.
**Upstream dependency:** Phase 2

When run as `__main__`, executes `test_engine()` — a smoke test with 4 sample queries.
In production, `RecommendationEngine` is imported directly by Phase 4.

## Execution Order (per query)

```
RecommendationEngine.__init__()
  └── SentenceTransformer("all-MiniLM-L6-v2") loaded
  └── chromadb.PersistentClient("data/chromadb/").get_collection("ted_talks")

engine.recommend(query_text, user_emotions, top_k)
  ├── 1. embed query
  ├── 2. ChromaDB vector search
  ├── 3. hybrid re-ranking
  └── 4. return top_k results
```

## Data Flow

### Step 1 — Embed Query
```
query_text  (raw user string)
        │
        ▼  SentenceTransformer.encode([query_text])
        query_vec   shape: (384,)  float list
```

### Step 2 — ChromaDB Vector Search
```
query_vec
        │
        ▼  collection.query(
               query_embeddings=[query_vec],
               n_results=min(top_k × 6, 3957),
               include=["documents", "metadatas", "distances"]
           )
        Candidates: list of (metadata_dict, cosine_distance)
        ChromaDB uses HNSW approximate nearest-neighbor search
        Distance metric: cosine  (stored as 1 − cosine_similarity)
```

### Step 3 — Hybrid Re-ranking
```
For each candidate:

  cosine_sim = 1.0 − chroma_distance   (converts back to similarity)

  _sentiment_alignment(talk_meta, user_emotions)
    For each of the 7 emotions:
      talk_val = float(meta[emotion])
      user_val = user_emotions[emotion]
      score    = 1.0 − |user_val − talk_val|
    sentiment_score = mean of all 7 scores
    (returns 1.0 if user_emotions is empty)

  combined_score = 0.7 × cosine_sim + 0.3 × sentiment_score

Sort all candidates by combined_score descending
Return candidates[:top_k]
```

### Step 4 — Output per Result
```
{
  title, speaker, url, description, views, duration,
  polarity, polarity_score,
  emotions: { anger, disgust, fear, joy, neutral, sadness, surprise },
  cosine_sim, sentiment_alignment, combined_score
}
```

---

---

# Phase 4 — Streamlit Chatbot (Live Inference)
### `src/phase4_chatbot.py`

**Input:** Live user text from Streamlit chat input
**Imports:** `RecommendationEngine` from `phase3_recommendation.py`
**Upstream dependency:** Phase 3 (and therefore Phase 2's ChromaDB)

Runs as a persistent Streamlit server. Each user message triggers the full
inference loop below.

## Model Loaded at Startup

| Model | Task | Cached via |
|---|---|---|
| `j-hartmann/emotion-english-distilroberta-base` | Real-time 7-class emotion classification | `@st.cache_resource` |
| `RecommendationEngine` (MiniLM + ChromaDB) | Semantic retrieval + re-ranking | `@st.cache_resource` |

## Execution Order (per user message)

```
User types → st.chat_input()
  │
  ├── Step 1 — analyze_emotions()
  ├── Step 2 — update_emotion_context()
  ├── Step 3 — engine.recommend()           (Phase 3 hybrid retrieval)
  ├── Step 4 — build_bot_response()
  └── Step 5 — st.markdown() + st.rerun()
```

## Data Flow

### Step 1 — `analyze_emotions(pipe, user_input)`
```
user_input  (raw text, truncated to 256 tokens internally)
        │
        ▼  j-hartmann DistilRoBERTa pipeline (top_k=None)
        { joy: 0.72, sadness: 0.05, anger: 0.01,
          fear: 0.04, neutral: 0.12, disgust: 0.03, surprise: 0.03 }
        (scores round to 4 decimal places)
```

### Step 2 — `update_emotion_context(context, current, alpha=0.4)`
```
current_emotions  (Step 1 output)
session_context   (st.session_state.emotion_context from previous turns)
        │
        ▼  For each of 7 emotions:
             updated = 0.4 × current + 0.6 × previous
        Smoothed emotion dict stored back into st.session_state.emotion_context
        (Tracks emotional drift across the full conversation)
```

### Step 3 — `engine.recommend(query_text, user_emotions, top_k)`
```
user_input + smoothed emotion dict
        │
        ▼  Phase 3 hybrid retrieval pipeline
           (see Phase 3 data flow above)
        List of top_k ranked talk dicts
```

### Step 4 — `build_bot_response(user_text, emotion_context, recommendations)`
```
Smoothed emotion dict → dominant emotion identified (argmax)
Ranked talk list
        │
        ▼  format_recommendation() per talk:
             title (hyperlinked), speaker, duration (mm:ss),
             views (comma-formatted), mood label + emoji,
             match % (combined_score × 100),
             description snippet (≤150 chars)
        │
        ▼  Assembled into a single Markdown string
        Response string → st.markdown()
        Sidebar: emotion progress bars re-rendered from session_context
        st.session_state.messages updated with user + assistant turns
        st.rerun() → Streamlit rerenders the full chat history
```

---

## End-to-End Data Flow Summary

```
ted_talks_en.csv  (raw)
        │
        ▼  Phase 1: clean, feature engineer
        data/cleaned_ted_talks.csv
        │
        ▼  Phase 2: DistilBERT polarity + RoBERTa emotions + MiniLM embeddings
        data/enriched_ted_talks.csv
        data/chromadb/  (ted_talks, 3,957 entries, 384-dim cosine index)
        │
        ▼  Phase 3: RecommendationEngine
           cosine_sim (0.7) + sentiment_alignment (0.3) hybrid ranking
        │
        ▼  Phase 4: Streamlit chatbot
           DistilRoBERTa → EMA emotion context → engine.recommend() → Markdown response
```

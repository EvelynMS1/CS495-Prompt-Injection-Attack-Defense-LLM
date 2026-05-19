# TED Talk Sentiment-Based Recommender Chatbot

## Project Description

An intelligent chatbot that recommends TED Talks by analyzing the sentiment of both the talk transcripts and the user's real-time prompts. By aligning the emotional tone of the user with the content of the talks, the system provides a more personalized and contextually relevant discovery experience.

## Objectives

- Process the TED Talk dataset with a multi-model sentiment analysis pipeline (DistilBERT, RoBERTa-GoEmotions)
- Embed talks using `all-MiniLM-L6-v2` and store vectors in ChromaDB
- Rank recommendations with a hybrid algorithm: cosine similarity (content) + sentiment alignment (mood match)
- Deliver a Streamlit chatbot that tracks the user's emotional context across the conversation

## Tools / Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Sentiment Analysis | DistilBERT (polarity), RoBERTa-GoEmotions (7 emotions) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Data | Pandas / NumPy |
| Frontend | Streamlit |
| ML Framework | PyTorch / Hugging Face Transformers |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (first time)

```bash
py -3.12 src/run_pipeline.py
```

This runs Phase 1 → Phase 2 → Phase 3 smoke test, then launches the Streamlit chatbot.

### 3. Launch the chatbot directly (after pipeline has run once)

```bash
py -3.12 -m streamlit run src/phase4_chatbot.py
```

### Run individual phases

```bash
py -3.12 src/phase1_data_cleaning.py        # EDA + clean data
py -3.12 src/phase2_sentiment_embeddings.py  # Sentiment scoring + ChromaDB
py -3.12 src/phase3_recommendation.py        # Smoke-test the engine
```

### Data

Place the TED Talk dataset at:

```
data/data/2020-05-01/ted_talks_en.csv
```

The dataset is excluded from version control (`.gitignore`). Download it from [TED Talks dataset on Kaggle](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset).

## Pipeline Overview

```
Phase 1  →  cleaned_ted_talks.csv  +  EDA plots
Phase 2  →  enriched_ted_talks.csv  +  ChromaDB (3,957 talks)
Phase 3  →  Recommendation engine (hybrid cosine + sentiment)
Phase 4  →  Streamlit chatbot with live mood tracking
```

## Team Members

- Evelyn Montes

**Attribution Requirement**:\
Any academic, research, or commercial usage must cite the original repository and authors.

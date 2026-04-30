# PLAN.md

## Project Overiew

**Title:** Video recommendation chatbot utilizing sentiment analysis
**Author:** Evelyn Montes
**Date:** April 27,2026

### Description

This project aims to build an intelligent chatbot that recommends TED talks by analyzing the sentiment of both the talk transcripts and the user's real-time prompts. By aligning the emotional tone of the user with the content of the talks, the system provides a more personalized and contextually relevant discovery experience.

### Objective

-Process the Ted Talk dataset utilizing a sentiment analysis model DistilBert, RoBERTa, GoEmotions, Bert
-Embedd the Ted talk dataset with latest embedding model in vectors
-Store vectors in a vector database Chroma

## Environment Setup

### Requirements

-Python 3.13
-Poetry for dependency management
-Virtual environment (.venv)-

### Quick Start (macOS/Linux)

```bash
git clone https://github.com/EvelynMS1/CS495-Prompt-Injection-Attack-Defense-LLM.git
cd project
python3.13 -m venv .venv
source .venv/bin/activate
poetry install # or: pip install -r requirements.txt
make test # verify everything works
```

### Quick Start

```powershell
git clone https://github.com/EvelynMS1/CS495-Prompt-Injection-Attack-Defense-LLM.git
cd project
python -m venv .venv
.venv\Scripts\Activate.ps1
poetry install
```

### Quick Start (Windows)

•

```powershell
git clone https://github.com/EvelynMS1/CS495-Prompt-Injection-Attack-Defense-LLM.git
cd project
python -m venv .venv
.venv\Scripts\Activate.ps1
poetry install
```

### Makefile Alternative (Windows)

If `make` is not available, use the scripts:

```powershell
python -m pytest tests/
python -m ruff check .
python src/main.py
```

## Tasks

## Data Source

- Path: ./data/ted_talks_en.csv
- Focus: English transcripts only

### Phase 1: Data Collection & Cleaning

**Goal:** Establish a robust development environment and prepare the raw data for processing.

- [ ] Import the TED Talk dataset.
- [ ] Perform comprehensive EDA to assess feature sparsity, identify data distribution across metadata fields (e.g., speaker occupations, topics), and establish preprocessing pipelines for noise reduction in text and numerical outliers.
- [ ] Handle missing values and outliers
- [ ] Develop a multi-dimensional feature set by vectorizing high-context metadata (Titles, Descriptions, Topics) and encoding categorical attributes (Speaker Occupations) to serve as weighting factors in the recommendation ranking algorithm.

### Phase 2: Data Engineering & Dataset Sentiment Analysis

**Goal:** Enrich the static dataset with sentiment scores.

- [ ] Implement a Sentiment Analysis pipeline (e.g., using a pre-trained BERT model DistilBERT and RoBERTa and GoEmotions) to score each talk for emotional testing.
- [ ] Identify emotion scores that each trasncript showcases, a semtiment trajectory that utilizes re-trained BERT model DistilBERT and RoBERTa and GoEmotions.
- [ ] Synthesize sentiment polarity scores with content embeddings to align recommendations with the user’s emotional intent and topical interests
- [ ] Save the enriched dataset into a structured format (vector database).

### Phase 3: Recommendation Engine Development Deployment

- [ ] Initialize ChromaDB with a collection schema that stores both semantic embeddings (content) and metadata (sentiment scores/trajectories)
- [ ] Develop a hybrid recommendation algorithm: - **cosine similarity** and **Sentiment alignment**.
- [ ] Test the engine with sample queries to ensure diverse and relevant outputs track mood matching, based on the user mood what is the recommendation mood.
- [ ] Add interactive prediction form
- [ ] Write documentation and README

## Phase 4: Chatbot Architecture & Live Prompt Processing

**Goal:** Build the interface and the real-time analysis pipeline.

- [ ] Design the chatbot conversation flow (Greeting -> Query -> Analysis -> Recommendation).
- [ ] Integrate a real-time sentiment analyzer for user prompts.
- [ ] Implement "Sentiment Context Tracking" to understand if a user's mood changes during the conversation.
- [ ] Connect the chatbot interface (Streamlit, Flask, or FastAPI) to the recommendation engine.

## Technical Stack

- **Language:** Python
- **ML Frameworks:** PyTorch / Hugging Face Transformers
- **Sentiment Analysis:** DistilBERT, BERT, GoEmotions
- **Data Handling:** Pandas / NumPy
- **Frontend:** Streamlit or a dedicated Web API
  """

### Data Processing

-Embedding

### Machine Learning Models

-Cosine Similarity

### Evaluation Metrics

-Sentiment Closeness
-Hallucination Rate

### Visualization

-interactive dashboard chatbot

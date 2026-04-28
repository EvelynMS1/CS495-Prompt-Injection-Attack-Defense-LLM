# PLAN.md

## Project Overiew

**Title:** Video recommendation chatbot utilizing sentiment analysis
**Author:** Evelyn Montes
**Date:** April 27,2026

### Description

This project aims to build an intelligent chatbot that recommends TED talks by analyzing the sentiment of both the talk transcripts and the user's real-time prompts. By aligning the emotional tone of the user with the content of the talks, the system provides a more personalized and contextually relevant discovery experience.

### Objective

-Process the Ted Talk dataset utilizing a sentiment analysis model DistilBert
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

### Phase 1: Data Collection & Cleaning

**Goal:** Establish a robust development environment and prepare the raw data for processing. -[ ] Import the TED Talk dataset (transcripts, metadata, and tags). -[ ] Perform Exploratory Data Analysis (EDA) to identify data distribution and cleaning requirements. -[ ] Handle missing values and outliers -[ ] Feature engineering (create new columns)

### Phase 2: Data Engineering & Dataset Sentiment Analysis

**Goal:** Enrich the static dataset with sentiment scores. -[ ] Implement a Sentiment Analysis pipeline (e.g., using a pre-trained BERT model or VADER) to score each talk. -[ ] Segment scores into categories (e.g., Inspiring, Informative, Challenging, Humorous). -[ ] Save the enriched dataset into a structured format (CSV or vector database).

### Phase 3: Recommendation Engine DevelopmentDeployment

**Goal:** Create the logic that matches users to content. -[ ] Implement text vectorization (e.g., TF-IDF or Word/Sentence Embeddings). -[ ] Develop a hybrid recommendation algorithm: - **Content Filtering:** Matching keywords and tags. - **Sentiment Matching:** Adjusting recommendations based on the emotional delta between user and talk. -[ ] Test the engine with sample queries to ensure diverse and relevant outputs. -[ ] Add interactive prediction form -[ ] Write documentation and README

## Phase 4: Chatbot Architecture & Live Prompt Processing

**Goal:** Build the interface and the real-time analysis pipeline.

- [ ] Design the chatbot conversation flow (Greeting -> Query -> Analysis -> Recommendation).
- [ ] Integrate a real-time sentiment analyzer for user prompts.
- [ ] Implement "Sentiment Context Tracking" to understand if a user's mood changes during the conversation.
- [ ] Connect the chatbot interface (Streamlit, Flask, or FastAPI) to the recommendation engine.

## Technical Stack

- **Language:** Python
- **ML Frameworks:** PyTorch / Hugging Face Transformers
- **Sentiment Analysis:** VADER or DistilBERT
- **Data Handling:** Pandas / NumPy
- **Frontend:** Streamlit or a dedicated Web API
  """

### Data Processing

-Embedding

### Machine Learning Models

-Cosine Similarity

### Evaluation Metrics

### Visualization

-interactive dashboard chatbot

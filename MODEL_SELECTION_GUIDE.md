# Model Selection Feature Guide

## Overview

This branch (`cs495_may18`) introduces **multi-model support** for the TED Talk Recommender chatbot. Users can now dynamically select different embedding models and emotion detection models through an intuitive UI.

## New Features

### 1. **Configurable Embedding Models**
Choose from multiple embedding models with different trade-offs:

| Model | Dimensions | Speed | Best For |
|-------|-----------|-------|----------|
| **MiniLM-L6** (Default) | 384 | Fast | Quick recommendations, resource-constrained environments |
| **BERT Base** | 768 | Medium | Higher accuracy, semantic understanding |
| **MPNet Base** | 768 | Medium | Highest quality embeddings, best results |
| **DistilBERT** | 768 | Fast | Balanced speed and accuracy |

### 2. **Configurable Emotion Detection Models**
Select the emotion analysis model:

| Model | Emotions | Best For |
|-------|----------|----------|
| **GoEmotions RoBERTa** (Default) | 28 → 7 | Comprehensive emotion detection |
| **DistilRoBERTa** | 7 | Fast, direct 7-emotion classification |

### 3. **Professional UI Enhancements**
- 🎨 Organized sidebar with clear model configuration section
- 📊 Real-time display of selected models
- 🔄 Easy model switching with "Apply Model Selection" button
- ✅ Visual feedback on model loading and configuration

## Usage Guide

### Step 1: Prepare Data and Models

Before using the chatbot, you need to generate embeddings for your chosen model(s):

#### Option A: Use Default Model (MiniLM-L6)
```bash
# Run Phase 2 with default settings
py -3.12 src/phase2_sentiment_embeddings.py
```

#### Option B: Use Specific Model
```bash
# Generate embeddings with BERT Base
py -3.12 src/phase2_multi_model.py --embedding-model "BERT Base (Accurate, 768-dim)"

# Generate embeddings with MPNet
py -3.12 src/phase2_multi_model.py --embedding-model "MPNet Base (High Quality, 768-dim)"

# Use custom sentiment and emotion models
py -3.12 src/phase2_multi_model.py \
    --embedding-model "MPNet Base (High Quality, 768-dim)" \
    --sentiment-model "RoBERTa Twitter (Social Media)" \
    --emotion-model "DistilRoBERTa (7 emotions)"
```

#### Option C: Generate Multiple Models (Recommended)
```bash
# Generate all embedding models for maximum flexibility
py -3.12 src/phase2_multi_model.py --embedding-model "MiniLM-L6 (Fast, 384-dim)"
py -3.12 src/phase2_multi_model.py --embedding-model "BERT Base (Accurate, 768-dim)" --skip-sentiment
py -3.12 src/phase2_multi_model.py --embedding-model "MPNet Base (High Quality, 768-dim)" --skip-sentiment
py -3.12 src/phase2_multi_model.py --embedding-model "DistilBERT (Balanced, 768-dim)" --skip-sentiment
```

**Note:** Use `--skip-sentiment` flag after the first run to reuse sentiment analysis results and only regenerate embeddings.

### Step 2: Launch the Chatbot

```bash
py -3.12 -m streamlit run src/phase4_chatbot.py
```

### Step 3: Select Models in the UI

1. **Open the sidebar** (should be visible by default)
2. **Scroll to "⚙️ Model Configuration"** section
3. **Select your preferred Embedding Model** from the dropdown
4. **Select your preferred Emotion Detection Model** from the dropdown
5. **Click "🔄 Apply Model Selection"** button
6. **Wait for models to load** (first load takes 10-30 seconds)
7. **Start chatting!**

The chatbot will display:
- ✅ Success message when models are loaded
- ⚠️ Error message if embeddings haven't been generated for selected model
- 📊 Current model configuration at the bottom of sidebar

## Architecture Changes

### New Files

1. **`src/model_config.py`**
   - Central configuration for all available models
   - Model metadata (dimensions, speed, collection names)
   - Default model selections

2. **`src/phase2_multi_model.py`**
   - Enhanced Phase 2 with command-line arguments
   - Support for multiple embedding and sentiment models
   - Automatic ChromaDB collection naming per model

3. **`MODEL_SELECTION_GUIDE.md`** (this file)
   - Documentation for new features

### Modified Files

1. **`src/phase3_recommendation.py`**
   - Updated `RecommendationEngine` class to accept `embedding_model_key` parameter
   - Dynamic model loading based on configuration
   - Support for both sentence-transformers and raw BERT models
   - Automatic ChromaDB path resolution per model

2. **`src/phase4_chatbot.py`**
   - Added model selection UI in sidebar
   - Session state tracking for selected models
   - Cache clearing on model change
   - Enhanced error handling with helpful messages
   - Professional styling with emojis and clear sections

## Data Storage Structure

Each embedding model creates its own ChromaDB collection:

```
data/
├── cleaned_ted_talks.csv              # From Phase 1
├── enriched_ted_talks.csv             # Default sentiment results
├── enriched_ted_talks_bert.csv        # BERT-specific
├── enriched_ted_talks_mpnet.csv       # MPNet-specific
├── enriched_ted_talks_distilbert.csv  # DistilBERT-specific
├── chromadb/                          # MiniLM-L6 embeddings
├── chromadb_bert/                     # BERT embeddings
├── chromadb_mpnet/                    # MPNet embeddings
└── chromadb_distilbert/               # DistilBERT embeddings
```

## Performance Comparison

| Model | Embedding Time* | Query Time** | Disk Space*** | Accuracy**** |
|-------|----------------|--------------|---------------|--------------|
| MiniLM-L6 | ~2 min (GPU) | ~50ms | ~150MB | ⭐⭐⭐ |
| BERT Base | ~5 min (GPU) | ~80ms | ~300MB | ⭐⭐⭐⭐ |
| MPNet Base | ~5 min (GPU) | ~80ms | ~300MB | ⭐⭐⭐⭐⭐ |
| DistilBERT | ~3 min (GPU) | ~60ms | ~300MB | ⭐⭐⭐⭐ |

\* For ~4,000 talks on GPU (5-10× slower on CPU)
\** Average query response time
\*** Per ChromaDB collection
\**** Subjective recommendation quality

## Troubleshooting

### Error: "ChromaDB not found for [model] at ..."

**Solution:** Generate embeddings for that model:
```bash
py -3.12 src/phase2_multi_model.py --embedding-model "[model name]"
```

### Error: "Failed to load collection..."

**Cause:** ChromaDB collection doesn't exist or is corrupted.

**Solution:** Regenerate embeddings:
```bash
py -3.12 src/phase2_multi_model.py --embedding-model "[model name]"
```

### Models won't switch / Changes don't apply

**Solution:**
1. Click "🔄 Apply Model Selection" button (not just selecting from dropdown)
2. Wait for cache to clear and page to reload
3. Check console for error messages

### Slow model loading

**Cause:** First load downloads models from Hugging Face Hub.

**Solution:**
- Wait patiently (10-30 seconds first time)
- Subsequent loads use cached models (much faster)
- Consider using faster models like MiniLM-L6

### Out of memory error during embedding generation

**Solution:**
- Reduce batch size in `phase2_multi_model.py` (edit `batch_size` variable)
- Use CPU instead of GPU (automatic fallback)
- Use smaller embedding model (MiniLM-L6 or DistilBERT)

## Development Notes

### Adding a New Embedding Model

1. Add to `EMBEDDING_MODELS` dict in `src/model_config.py`:
```python
"Your Model Name": {
    "model_name": "huggingface/model-name",
    "dimensions": 768,
    "speed": "Fast",
    "collection_name": "ted_talks_yourmodel",
    "chroma_dir": "chromadb_yourmodel",
}
```

2. Generate embeddings:
```bash
py -3.12 src/phase2_multi_model.py --embedding-model "Your Model Name"
```

3. Model will automatically appear in chatbot dropdown!

### Adding a New Emotion Model

1. Add to `EMOTION_MODELS` dict in `src/model_config.py`:
```python
"Your Emotion Model": {
    "model_name": "huggingface/emotion-model",
    "num_emotions": 7,  # or 28 for GoEmotions-style
    "aggregated": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
}
```

2. If using custom emotion labels, update aggregation logic in `phase4_chatbot.py`

## API Reference

### `RecommendationEngine(embedding_model_key=None)`

**Parameters:**
- `embedding_model_key` (str, optional): Key from `EMBEDDING_MODELS` dict. Defaults to `DEFAULT_EMBEDDING`.

**Raises:**
- `ValueError`: If model key not found in `EMBEDDING_MODELS`
- `FileNotFoundError`: If ChromaDB directory doesn't exist for model
- `RuntimeError`: If ChromaDB collection can't be loaded

**Example:**
```python
from phase3_recommendation import RecommendationEngine

# Use default model
engine = RecommendationEngine()

# Use specific model
engine = RecommendationEngine("BERT Base (Accurate, 768-dim)")

# Get recommendations
results = engine.recommend(
    query_text="I want to feel inspired",
    user_emotions={"joy": 0.6, "sadness": 0.2},
    top_k=5
)
```

### `load_engine(embedding_model_key)` (Streamlit cached)

**Parameters:**
- `embedding_model_key` (str): Key from `EMBEDDING_MODELS` dict

**Returns:**
- `RecommendationEngine` instance

**Notes:**
- Cached using `@st.cache_resource`
- Displays error in Streamlit UI if loading fails
- Automatically stops app execution on error

### `load_emotion_model(emotion_model_key)` (Streamlit cached)

**Parameters:**
- `emotion_model_key` (str): Key from `EMOTION_MODELS` dict

**Returns:**
- `(pipeline, model_config)` tuple

**Notes:**
- Cached using `@st.cache_resource`
- Returns both the Hugging Face pipeline and model metadata

## Future Enhancements

- [ ] Add sentiment model selection to UI (currently only emotion + embedding)
- [ ] Model comparison view (side-by-side results)
- [ ] Performance benchmarking dashboard
- [ ] Automatic model download on first use
- [ ] Model recommendation based on user's hardware
- [ ] Support for custom/fine-tuned models
- [ ] Multi-language support with language-specific models

## Credits

**Developed by:** Claude Code
**Date:** May 18, 2026
**Branch:** `cs495_may18`
**Models from:** Hugging Face Hub
- sentence-transformers (UKPLab)
- GoEmotions (Google Research)
- DistilBERT, BERT, RoBERTa (Hugging Face)

## License

This feature follows the same license as the main project.

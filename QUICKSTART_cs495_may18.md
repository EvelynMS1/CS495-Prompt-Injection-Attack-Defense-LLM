# Quick Start Guide - Branch cs495_may18

## What's New?

✨ **Multi-Model Support** - Choose from 4 embedding models and 2 emotion detection models
🎨 **Professional UI** - Organized sidebar with clear model configuration
📊 **Real-time Display** - See your current model configuration at all times
🔄 **Easy Switching** - Change models on-the-fly without restarting

## Step-by-Step Setup

### 1. Push to Remote (if not already done)

```bash
# You'll need to authenticate with GitHub
git push -u origin cs495_may18
```

If you get authentication errors, see `PUSH_INSTRUCTIONS.txt` for alternatives.

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install streamlit transformers sentence-transformers chromadb torch pandas numpy
```

### 3. Prepare Data (First Time Only)

```bash
# Step 3a: Run Phase 1 to clean data
python3 src/phase1_data_cleaning.py

# Step 3b: Run Phase 2 to generate embeddings (choose your model)
# Option 1: Default MiniLM (fastest, recommended for testing)
python3 src/phase2_multi_model.py --embedding-model "MiniLM-L6 (Fast, 384-dim)"

# Option 2: BERT (higher quality)
python3 src/phase2_multi_model.py --embedding-model "BERT Base (Accurate, 768-dim)"

# Option 3: MPNet (highest quality)
python3 src/phase2_multi_model.py --embedding-model "MPNet Base (High Quality, 768-dim)"
```

**Note:** Phase 2 takes 5-30 minutes depending on your hardware (faster with GPU).

### 4. Launch the Chatbot

```bash
python3 -m streamlit run src/phase4_chatbot.py
```

The chatbot will open in your browser at `http://localhost:8501`

### 5. Using the Model Selection UI

1. **Open the sidebar** (left panel)
2. **Scroll to "⚙️ Model Configuration"**
3. **Select your preferred models:**
   - Embedding Model (for content similarity)
   - Emotion Detection Model (for mood analysis)
4. **Click "🔄 Apply Model Selection"**
5. **Wait for models to load** (~10-30 seconds first time)
6. **Start chatting!**

## Testing Without Data

If you want to test the UI without generating embeddings:

```bash
# Run the test script
python3 test_installation.py
```

This will verify:
- ✅ All required packages are installed
- ✅ All project files exist
- ✅ Code has no syntax errors
- ⚠️ Data directory status

## Model Selection Options

### Embedding Models
| Model | Best For | Speed | Quality |
|-------|----------|-------|---------|
| MiniLM-L6 | Quick testing, low resources | ⚡⚡⚡ | ⭐⭐⭐ |
| BERT Base | Balanced accuracy | ⚡⚡ | ⭐⭐⭐⭐ |
| MPNet Base | Best results | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| DistilBERT | Speed + accuracy | ⚡⚡⚡ | ⭐⭐⭐⭐ |

### Emotion Models
| Model | Emotions | Best For |
|-------|----------|----------|
| GoEmotions | 28→7 | Comprehensive detection |
| DistilRoBERTa | 7 | Fast, direct classification |

## Example Queries to Try

Once the chatbot is running:

- "I feel burnt out and need motivation"
- "Something funny about human behavior"
- "I am curious about AI and the future"
- "I'm feeling anxious about climate change"
- "Inspire me to be creative"

## Troubleshooting

### "ChromaDB not found for [model]"
**Fix:** Run Phase 2 with that model:
```bash
python3 src/phase2_multi_model.py --embedding-model "[model name]"
```

### "No module named 'streamlit'"
**Fix:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Models won't load/switch
**Fix:** Click the "🔄 Apply Model Selection" button (not just the dropdown)

### Out of memory error
**Fix:** Use MiniLM-L6 model (smallest, fastest)

## File Structure

```
📁 cs495_may18/
├── 📄 src/model_config.py          [NEW] Model definitions
├── 📄 src/phase2_multi_model.py    [NEW] Multi-model Phase 2
├── 📝 src/phase3_recommendation.py [MOD] Dynamic model loading
├── 🎨 src/phase4_chatbot.py        [MOD] Model selection UI
├── 📚 MODEL_SELECTION_GUIDE.md     [NEW] Full documentation
├── 📊 FLOWCHART_PIPELINE.md        [NEW] Pipeline diagram
├── 🧪 test_installation.py         [NEW] Test script
└── 📋 QUICKSTART_cs495_may18.md    [NEW] This file
```

## Performance Notes

- **First model load:** 10-30 seconds (downloads from Hugging Face)
- **Subsequent loads:** 2-5 seconds (uses cached models)
- **Query time:** 50-150ms per recommendation
- **Switching models:** Clears cache, reloads (~5-10 seconds)

## What to Demo

1. **Model Selection UI** - Show the clean, professional sidebar
2. **Multiple Models** - Switch between MiniLM and BERT to show different results
3. **Emotion Detection** - Show how your emotional profile updates as you chat
4. **Real-time Configuration** - Point out the "Current Configuration" display
5. **Error Handling** - Try selecting a model without embeddings (shows helpful error)

## Support

- **Full Documentation:** See `MODEL_SELECTION_GUIDE.md`
- **Pipeline Overview:** See `FLOWCHART_PIPELINE.md`
- **Test Installation:** Run `python3 test_installation.py`

## Credits

**Branch:** cs495_may18
**Date:** May 18, 2026
**Created by:** Claude Code
**Features:** Multi-model support, professional UI, dynamic configuration

---

**Ready to start?** Run the chatbot and explore! 🚀

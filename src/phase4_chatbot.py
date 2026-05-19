"""
Phase 4: Streamlit Chatbot
Flow: Greeting -> User query -> Real-time sentiment analysis -> Recommendation -> Display
Tracks emotional context across the conversation.
Supports multiple embedding and emotion detection models.
"""

import os
import sys

import streamlit as st
from transformers import pipeline

sys.path.insert(0, os.path.dirname(__file__))
from phase3_recommendation import RecommendationEngine
from cf_baseline import CFBiasBaseline
from model_config import (
    EMBEDDING_MODELS,
    EMOTION_MODELS,
    DEFAULT_EMBEDDING,
    DEFAULT_EMOTION,
    EMOTION_COLS,
    GO_EMOTIONS_MAP,
)

EMOTION_LABELS = {
    "joy": "Joy",
    "sadness": "Sadness",
    "anger": "Anger",
    "fear": "Fear",
    "neutral": "Neutral",
    "disgust": "Disgust",
    "surprise": "Surprise",
}
EMOTION_EMOJI = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "neutral": "😐",
    "disgust": "🤢",
    "surprise": "😲",
}


# -- Cached resources ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model(emotion_model_key):
    """Load emotion detection model based on user selection."""
    if emotion_model_key not in EMOTION_MODELS:
        emotion_model_key = DEFAULT_EMOTION

    model_config = EMOTION_MODELS[emotion_model_key]
    model_name = model_config["model_name"]

    return pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        truncation=True,
        max_length=256,
    ), model_config


@st.cache_resource(show_spinner="Connecting to recommendation engine...")
def load_engine(embedding_model_key):
    """Load recommendation engine with specified embedding model."""
    try:
        return RecommendationEngine(embedding_model_key)
    except (FileNotFoundError, RuntimeError) as e:
        st.error(f"**Error loading recommendation engine:**\n\n{str(e)}")
        st.info(
            "💡 **Tip:** Run Phase 2 to generate embeddings for this model:\n"
            f"```bash\npy -3.12 src/phase2_sentiment_embeddings.py\n```"
        )
        st.stop()


@st.cache_resource(show_spinner="Loading CF bias baseline...")
def load_cf_baseline():
    """Load CF bias baseline for comparison."""
    return CFBiasBaseline()


# -- Helpers ------------------------------------------------------------------

def analyze_emotions(pipe, text, model_config):
    """Analyze emotions with support for different emotion models."""
    results = pipe(text)[0]

    # If using GoEmotions (28 labels), aggregate to 7
    if model_config["num_emotions"] == 28:
        aggregated = {e: 0.0 for e in EMOTION_COLS}
        for r in results:
            broad = GO_EMOTIONS_MAP.get(r["label"], "neutral")
            aggregated[broad] = round(aggregated[broad] + r["score"], 4)
        # Normalize
        total = sum(aggregated.values()) or 1.0
        return {k: round(v / total, 4) for k, v in aggregated.items()}
    else:
        # Already 7 emotions
        return {r["label"]: round(r["score"], 4) for r in results}


def update_emotion_context(context, current, alpha=0.4):
    updated = {}
    for emotion in EMOTION_COLS:
        prev = context.get(emotion, 0.0)
        curr = current.get(emotion, 0.0)
        updated[emotion] = round(alpha * curr + (1 - alpha) * prev, 4)
    return updated


def format_recommendation(rec, rank):
    mins = rec["duration"] // 60
    secs = rec["duration"] % 60
    dominant_emotion = max(rec["emotions"], key=rec["emotions"].get)
    emoji = EMOTION_EMOJI.get(dominant_emotion, "")
    match_pct = int(rec["combined_score"] * 100)

    lines = [
        f"**{rank}. [{rec['title']}]({rec['url']})**",
        f"*{rec['speaker']}* &nbsp;|&nbsp; {mins}m {secs:02d}s &nbsp;|&nbsp; {rec['views']:,} views",
        f"Mood: {rec['polarity'].title()} {emoji} &nbsp;|&nbsp; Match: {match_pct}%",
    ]
    if rec.get("description"):
        snippet = rec["description"][:150].rstrip()
        if len(rec["description"]) > 150:
            snippet += "..."
        lines.append(f"> {snippet}")
    return "\n".join(lines)


def build_bot_response(user_text, emotion_context, recommendations):
    top_emotion = max(emotion_context, key=emotion_context.get)
    emoji = EMOTION_EMOJI.get(top_emotion, "")

    parts = [
        f"I sense **{EMOTION_LABELS.get(top_emotion, top_emotion)}** {emoji} in your message.",
        "Here are TED Talks matched to your mood and topic:\n",
    ]
    for i, rec in enumerate(recommendations, 1):
        parts.append(format_recommendation(rec, i))
        parts.append("")

    parts.append("---")
    parts.append("*Keep chatting — I track your emotional context across the conversation.*")
    return "\n".join(parts)


# -- UI -----------------------------------------------------------------------

st.set_page_config(
    page_title="TED Talk Recommender",
    page_icon="🎤",
    layout="wide",
)

st.title("🎤 TED Talk Recommender")
st.caption(
    "AI-powered recommendation system with configurable embedding and emotion models"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your TED Talk recommender powered by sentiment analysis.\n\n"
                "**First, please select your preferred models from the sidebar.**\n\n"
                "Then tell me how you are feeling, or describe a topic you want to explore. "
                "For example:\n"
                "- *I feel burnt out and need motivation*\n"
                "- *Something funny about human behavior*\n"
                "- *I am curious about AI and the future*"
            ),
        }
    ]
if "emotion_context" not in st.session_state:
    st.session_state.emotion_context = {}
if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = DEFAULT_EMBEDDING
if "selected_emotion_model" not in st.session_state:
    st.session_state.selected_emotion_model = DEFAULT_EMOTION

# -- Sidebar: Model Configuration ---------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")

    # Embedding Model Selection
    st.subheader("1️⃣ Embedding Model")
    embedding_options = list(EMBEDDING_MODELS.keys())

    selected_embedding = st.selectbox(
        "Choose embedding model:",
        options=embedding_options,
        index=embedding_options.index(st.session_state.selected_embedding_model),
        help="Determines how talk content is vectorized for similarity search",
        key="embedding_selector"
    )

    # Show model details
    if selected_embedding:
        model_info = EMBEDDING_MODELS[selected_embedding]
        st.caption(
            f"🔹 **Dimensions:** {model_info['dimensions']} | "
            f"**Speed:** {model_info['speed']}"
        )

    st.divider()

    # Emotion Model Selection
    st.subheader("2️⃣ Emotion Detection Model")
    emotion_options = list(EMOTION_MODELS.keys())

    selected_emotion = st.selectbox(
        "Choose emotion model:",
        options=emotion_options,
        index=emotion_options.index(st.session_state.selected_emotion_model),
        help="Analyzes emotional tone of your queries in real-time",
        key="emotion_selector"
    )

    # Show model details
    if selected_emotion:
        emotion_info = EMOTION_MODELS[selected_emotion]
        st.caption(
            f"🔹 **Emotions:** {emotion_info['num_emotions']} labels"
        )

    st.divider()

    # Apply button
    if st.button("🔄 Apply Model Selection", use_container_width=True, type="primary"):
        if (selected_embedding != st.session_state.selected_embedding_model or
            selected_emotion != st.session_state.selected_emotion_model):

            st.session_state.selected_embedding_model = selected_embedding
            st.session_state.selected_emotion_model = selected_emotion

            # Clear cache and reset conversation
            st.cache_resource.clear()
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.emotion_context = {}

            st.success("✅ Models updated! Reloading...")
            st.rerun()
        else:
            st.info("Models already selected")

# Load models with current selection
emotion_pipe, emotion_config = load_emotion_model(st.session_state.selected_emotion_model)
engine = load_engine(st.session_state.selected_embedding_model)
cf_baseline = load_cf_baseline()

# Continue sidebar
with st.sidebar:
    # Emotional Profile
    st.header("❤️ Your Emotional Profile")
    st.caption("Updates as you chat")

    if st.session_state.emotion_context:
        sorted_emotions = sorted(
            st.session_state.emotion_context.items(), key=lambda x: -x[1]
        )
        for emotion, score in sorted_emotions:
            label = f"{EMOTION_EMOJI.get(emotion, '')} {EMOTION_LABELS.get(emotion, emotion)}"
            st.progress(score, text=f"{label}: {score:.0%}")
    else:
        st.info("Start chatting to see your emotional profile here.")

    st.divider()

    # Settings
    st.header("🎚️ Settings")
    top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.emotion_context = {}
        st.rerun()

    # Model info display
    st.divider()
    st.caption("**Current Configuration:**")
    st.caption(f"📊 Embedding: {st.session_state.selected_embedding_model}")
    st.caption(f"💭 Emotion: {st.session_state.selected_emotion_model}")

# -- Chat history -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -- User input ---------------------------------------------------------------
if user_input := st.chat_input("How are you feeling? What are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your mood and finding talks..."):
            current_emotions = analyze_emotions(emotion_pipe, user_input, emotion_config)
            st.session_state.emotion_context = update_emotion_context(
                st.session_state.emotion_context, current_emotions
            )
            recommendations = engine.recommend(
                query_text=user_input,
                user_emotions=st.session_state.emotion_context,
                top_k=top_k,
            )
            response = build_bot_response(
                user_input, st.session_state.emotion_context, recommendations
            )

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

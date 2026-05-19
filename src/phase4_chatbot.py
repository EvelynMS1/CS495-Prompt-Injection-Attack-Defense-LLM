"""
Phase 4: Streamlit Chatbot
Flow: Greeting -> User query -> Real-time sentiment analysis -> Recommendation -> Display
Tracks emotional context across the conversation.
"""

import os
import sys

import streamlit as st
from transformers import pipeline

sys.path.insert(0, os.path.dirname(__file__))
from phase3_recommendation import RecommendationEngine, EMOTION_COLS
from cf_baseline import CFBiasBaseline

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
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
def load_emotion_model():
    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=None,
        truncation=True,
        max_length=256,
    )


@st.cache_resource(show_spinner="Connecting to recommendation engine...")
def load_engine():
    return RecommendationEngine()


@st.cache_resource(show_spinner="Loading CF bias baseline...")
def load_cf_baseline():
    return CFBiasBaseline()


# -- Helpers ------------------------------------------------------------------

def analyze_emotions(pipe, text):
    results = pipe(text)[0]
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

st.title("TED Talk Recommender")
st.caption(
    "Describe what you are feeling or looking for, and I will find TED Talks aligned to your mood."
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your TED Talk recommender powered by sentiment analysis.\n\n"
                "Tell me how you are feeling, or describe a topic you want to explore. "
                "For example:\n"
                "- *I feel burnt out and need motivation*\n"
                "- *Something funny about human behavior*\n"
                "- *I am curious about AI and the future*"
            ),
        }
    ]
if "emotion_context" not in st.session_state:
    st.session_state.emotion_context = {}

emotion_pipe = load_emotion_model()
engine = load_engine()
cf_baseline = load_cf_baseline()

# -- Sidebar: emotional context -----------------------------------------------
with st.sidebar:
    st.header("Your Emotional Profile")
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
    top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.emotion_context = {}
        st.rerun()

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
            current_emotions = analyze_emotions(emotion_pipe, user_input)
            st.session_state.emotion_context = update_emotion_context(
                st.session_state.emotion_context, current_emotions
            )
            recommendations = engine.recommend(
                query_text=user_input,
                user_emotions=st.session_state.emotion_context,
                top_k=top_k,
            )
            cf_recs = cf_baseline.recommend(
                query_text=user_input,
                user_emotions=st.session_state.emotion_context,
                top_k=top_k,
            )
            response = build_bot_response(
                user_input, st.session_state.emotion_context, recommendations
            )

        st.markdown(response)

        with st.expander("Compare: CF Popularity Bias Baseline"):
            st.caption(
                "**CF item-bias baseline** — ranked by log-normalized view count "
                "(popularity) + the same emotion-alignment score as the main model. "
                "No semantic understanding of your query is used. "
                "Overlap with the results above shows what popularity alone can explain; "
                "divergence shows what semantic retrieval adds."
            )
            for i, rec in enumerate(cf_recs, 1):
                st.markdown(format_recommendation(rec, i))
                st.markdown("")

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

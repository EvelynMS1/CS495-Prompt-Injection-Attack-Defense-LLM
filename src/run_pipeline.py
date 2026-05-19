"""
Pipeline runner: Phase 1 -> Phase 2 -> Phase 3 test -> Phase 4 chatbot launch
Run from the project root: py -3.12 src/run_pipeline.py
"""

import os
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))


def run_phase(script_name, label):
    path = os.path.join(SRC, script_name)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"\nERROR: {label} failed (exit code {result.returncode}). Stopping.")
        raise SystemExit(result.returncode)
    print(f"\n{label} -- DONE")


def main():
    print("TED Talk Recommender -- Full Pipeline")
    print("=" * 60)

    run_phase("phase1_data_cleaning.py", "Phase 1: Data Cleaning & EDA")
    run_phase("phase2_sentiment_embeddings.py", "Phase 2: Sentiment Analysis & Embeddings")
    run_phase("phase3_recommendation.py", "Phase 3: Recommendation Engine (smoke test)")

    print(f"\n{'='*60}")
    print("  Phase 4: Launching Chatbot")
    print(f"{'='*60}")
    print("Starting Streamlit app... (press Ctrl+C to stop)\n")
    chatbot = os.path.join(SRC, "phase4_chatbot.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", chatbot])


if __name__ == "__main__":
    main()

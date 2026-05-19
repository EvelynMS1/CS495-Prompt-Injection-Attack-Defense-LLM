#!/usr/bin/env python3
"""
Test script to verify the new multi-model feature installation
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    required_modules = {
        "streamlit": "pip install streamlit",
        "transformers": "pip install transformers",
        "sentence_transformers": "pip install sentence-transformers",
        "chromadb": "pip install chromadb",
        "torch": "pip install torch",
        "pandas": "pip install pandas",
        "numpy": "pip install numpy",
    }

    missing = []
    for module, install_cmd in required_modules.items():
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - Install with: {install_cmd}")
            missing.append(module)

    print()
    return len(missing) == 0


def test_project_files():
    """Test that all new files exist."""
    print("=" * 60)
    print("Testing project files...")
    print("=" * 60)

    required_files = [
        "src/model_config.py",
        "src/phase3_recommendation.py",
        "src/phase4_chatbot.py",
        "src/phase2_multi_model.py",
        "MODEL_SELECTION_GUIDE.md",
        "FLOWCHART_PIPELINE.md",
    ]

    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath} - Missing!")
            all_exist = False

    print()
    return all_exist


def test_model_config():
    """Test that model_config.py is valid."""
    print("=" * 60)
    print("Testing model_config.py...")
    print("=" * 60)

    try:
        sys.path.insert(0, "src")
        from model_config import (
            EMBEDDING_MODELS,
            EMOTION_MODELS,
            SENTIMENT_MODELS,
            DEFAULT_EMBEDDING,
            DEFAULT_EMOTION,
            EMOTION_COLS,
        )

        print(f"✅ Module imports successfully")
        print(f"✅ Found {len(EMBEDDING_MODELS)} embedding models")
        print(f"✅ Found {len(EMOTION_MODELS)} emotion models")
        print(f"✅ Found {len(SENTIMENT_MODELS)} sentiment models")
        print(f"✅ Default embedding: {DEFAULT_EMBEDDING}")
        print(f"✅ Default emotion: {DEFAULT_EMOTION}")
        print(f"✅ Emotion columns: {len(EMOTION_COLS)}")
        print()
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        return False


def test_phase3_recommendation():
    """Test that phase3_recommendation.py can be imported."""
    print("=" * 60)
    print("Testing phase3_recommendation.py...")
    print("=" * 60)

    try:
        from phase3_recommendation import RecommendationEngine
        print("✅ RecommendationEngine class imports successfully")
        print("✅ Ready to instantiate with model selection")
        print()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        return False


def check_data_directory():
    """Check if data directory exists."""
    print("=" * 60)
    print("Checking data directory...")
    print("=" * 60)

    if os.path.exists("data"):
        print("✅ data/ directory exists")

        subdirs = ["chromadb", "chromadb_bert", "chromadb_mpnet", "chromadb_distilbert"]
        for subdir in subdirs:
            path = f"data/{subdir}"
            if os.path.exists(path):
                print(f"✅ {path}/ exists")
            else:
                print(f"⚠️  {path}/ not found (run Phase 2 to generate)")

        # Check for enriched CSVs
        if os.path.exists("data/cleaned_ted_talks.csv"):
            print("✅ data/cleaned_ted_talks.csv exists")
        else:
            print("⚠️  data/cleaned_ted_talks.csv not found (run Phase 1)")

    else:
        print("⚠️  data/ directory not found")
        print("   You need to run Phase 1 and Phase 2 before using the chatbot")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TED Talk Recommender - Multi-Model Feature Test")
    print("=" * 60)
    print()

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Project Files", test_project_files()))
    results.append(("Model Config", test_model_config()))
    results.append(("Phase 3 Recommendation", test_phase3_recommendation()))

    check_data_directory()

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")

    print()

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("✅ All tests passed! The multi-model feature is ready to use.")
        print()
        print("Next steps:")
        print("1. If data/ directory doesn't exist, run Phase 1:")
        print("   python3 src/phase1_data_cleaning.py")
        print()
        print("2. Generate embeddings with your preferred model:")
        print("   python3 src/phase2_multi_model.py --embedding-model \"MiniLM-L6 (Fast, 384-dim)\"")
        print()
        print("3. Launch the chatbot:")
        print("   python3 -m streamlit run src/phase4_chatbot.py")
        print()
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print()
        print("Common fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Ensure all files are in the correct location")
        print("- Check for syntax errors in modified files")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

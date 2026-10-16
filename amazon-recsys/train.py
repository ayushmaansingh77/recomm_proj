"""Build everything:  python train.py --category Video_Games   |   python train.py --synthetic"""
import argparse

from config import DEFAULT_CATEGORY
from src.pipeline import build_artifacts

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Which Amazon Reviews 2023 category to stream from HuggingFace
    p.add_argument("--category", default=DEFAULT_CATEGORY, help="Amazon Reviews 2023 category, e.g. Video_Games")

    # Limit how many review rows are streamed; metadata is always streamed in full then filtered
    p.add_argument("--max-reviews", type=int, default=400_000, help="reviews to stream from the raw file")

    # Held-out users for the cold-start benchmark; more = more accurate metrics, slower run
    p.add_argument("--eval-users", type=int, default=1000)

    # tfidf is lightweight and deployment-safe; sbert is richer but pulls in PyTorch
    p.add_argument("--embedder", choices=["tfidf", "sbert"], default="tfidf")

    # Skip the download entirely and use generated synthetic data instead
    p.add_argument("--synthetic", action="store_true", help="offline synthetic data (no download)")

    a = p.parse_args()

    # build_artifacts orchestrates: load → benchmark → refit → save
    _, results, _ = build_artifacts(a.category, a.synthetic, a.max_reviews, a.eval_users, a.embedder)
    print(results.round(4).to_string(index=False))

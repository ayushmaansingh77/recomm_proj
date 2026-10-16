from pathlib import Path

# Project root is the directory this file lives in
ROOT = Path(__file__).resolve().parent

# Raw downloaded data is cached here (Parquet); excluded from git
DATA_DIR = ROOT / "data"

# Trained models and benchmark results land here; committed to git for Streamlit Cloud
ARTIFACT_DIR = ROOT / "artifacts"

# Amazon Reviews 2023 (McAuley Lab) raw JSONL files hosted on Hugging Face
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"

# Category used when none is specified; change here to retrain on a different domain
DEFAULT_CATEGORY = "Video_Games"   # alternatives: Software, Digital_Music, All_Beauty, Handmade_Products

# Ratings at or above this threshold count as a positive ("liked") interaction
LIKE_THRESHOLD = 4.0

# Items with fewer than this many positive interactions are shown as "new" in the UI
NEW_ITEM_THRESHOLD = 5

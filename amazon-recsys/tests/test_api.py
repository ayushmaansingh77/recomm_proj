"""Test suite for RecSys Serving API and Algorithmic Interfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import ARTIFACT_DIR
from src.recommender import RecSys


@pytest.fixture(scope="module")
def recsys_instance() -> RecSys:
    """Fixture to load RecSys model once for all tests."""
    model_path = ARTIFACT_DIR / "recsys.joblib"
    if not model_path.exists():
        pytest.skip("recsys.joblib artifact not found. Run train.py or synthetic build first.")
    return RecSys.load(model_path)


def test_catalogue_schema(recsys_instance: RecSys):
    """Test that catalogue contains all expected columns and valid rows."""
    expected_cols = [
        "asin", "title", "category", "price", "image", "store",
        "average_rating", "rating_number", "description", "n_pos", "is_new"
    ]
    for col in expected_cols:
        assert col in recsys_instance.items.columns, f"Missing expected column '{col}' in catalogue"
    assert len(recsys_instance.items) > 0, "Catalogue has 0 items"


def test_weights_dynamic_blending(recsys_instance: RecSys):
    """Test adaptive weight shifts across cold, warming, and warm history regimes."""
    w0 = recsys_instance.weights(0)
    assert w0["Popularity"] > 0.8, "Cold user (n=0) should rely primarily on Popularity"
    assert w0["Content"] == 0.0 or w0["Content"] < 0.2
    assert w0["Collaborative"] == 0.0

    w1 = recsys_instance.weights(1)
    assert w1["Content"] >= w1["Popularity"], "At n=1 Content matches Popularity"

    w2 = recsys_instance.weights(2)
    assert w2["Content"] > w2["Popularity"], "At n=2 Content overtakes Popularity"

    w10 = recsys_instance.weights(10)
    assert w10["Collaborative"] > w1["Collaborative"], "Warm user (n=10) should have higher CF weight than n=1"



def test_recommend_cold_user(recsys_instance: RecSys):
    """Test recommendation output for cold user with zero history."""
    k = 8
    recs = recsys_instance.recommend(liked=[], seen=(), k=k, model="Hybrid (cold-aware)")
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == k
    assert "score" in recs.columns
    assert (recs["score"].to_numpy()[1:] <= recs["score"].to_numpy()[:-1]).all(), "Recommendations must be sorted descending by score"



def test_recommend_warm_user(recsys_instance: RecSys):
    """Test recommendation output for warm user with liked items excluded from results."""
    liked_items = [0, 1, 2]
    k = 6
    recs = recsys_instance.recommend(liked=liked_items, seen=liked_items, k=k, model="Hybrid")
    assert len(recs) == k
    for item_idx in liked_items:
        assert item_idx not in recs.index, f"Seen/liked item {item_idx} should be excluded from recommendations"


def test_recommend_category_filter(recsys_instance: RecSys):
    """Test filtering recommendations by category."""
    cats = recsys_instance.categories(top=5)
    if not cats:
        pytest.skip("No categories found in catalogue")
    target_cat = cats[0]
    recs = recsys_instance.recommend(liked=[0], k=5, category=target_cat)
    if not recs.empty:
        assert (recs["category"] == target_cat).all(), "All recommended items must match target category"


def test_recommend_cold_items_only(recsys_instance: RecSys):
    """Test filtering recommendations strictly for new/cold items."""
    recs = recsys_instance.recommend(liked=[0], k=5, only_new=True)
    if not recs.empty:
        assert (recs["is_new"] == True).all(), "All items must have is_new == True"


def test_similar_content(recsys_instance: RecSys):
    """Test finding nearest neighbours via content embedding space."""
    target_item = 0
    sim = recsys_instance.similar(target_item, model="Content", k=5)
    assert isinstance(sim, pd.DataFrame)
    assert len(sim) == 5
    assert target_item not in sim.index, "Target item must not be included in its own neighbours"


def test_similar_to_raw_text(recsys_instance: RecSys):
    """Test zero-shot natural language query simulation via TF-IDF vector projection."""
    query = "action adventure open world with sword combat"
    sim = recsys_instance.similar_to_text(query, k=5)
    assert isinstance(sim, pd.DataFrame)
    assert len(sim) == 5
    assert "score" in sim.columns


def test_search(recsys_instance: RecSys):
    """Test catalogue substring search."""
    res = recsys_instance.search("game", limit=5)
    assert isinstance(res, pd.DataFrame)
    assert len(res) <= 5
    for title in res["title"]:
        assert "game" in title.lower()


def test_onboarding_items(recsys_instance: RecSys):
    """Test onboarding questionnaire item generation."""
    picks = recsys_instance.onboarding_items(16)
    assert 10 <= len(picks) <= 16, f"Expected 10-16 onboarding items, got {len(picks)}"
    assert len(set(picks)) == len(picks), "Onboarding items must be unique"
    for p in picks:
        assert p in recsys_instance.items.index



def test_random_new_item(recsys_instance: RecSys):
    """Test sampling a cold item from the catalogue."""
    rng = np.random.default_rng(42)
    cold_item_id = recsys_instance.random_new_item(rng)
    assert cold_item_id in recsys_instance.items.index
    assert recsys_instance.items.at[cold_item_id, "is_new"] == True

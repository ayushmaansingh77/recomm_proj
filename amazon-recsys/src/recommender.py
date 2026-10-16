"""Serving layer used by the Streamlit app: wraps fitted models + catalogue."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import NEW_ITEM_THRESHOLD

SHOW_COLS = ["asin", "title", "category", "price", "image", "store",
             "average_rating", "rating_number", "description"]


class RecSys:
    def __init__(self, items: pd.DataFrame, models: dict, new_threshold: int = NEW_ITEM_THRESHOLD):
        self.items = items[SHOW_COLS].copy()
        self.items["description"] = self.items["description"].str[:400]
        self.models = models
        counts = models["Popularity"].counts.astype(int)
        self.items["n_pos"] = counts
        self.items["is_new"] = counts < new_threshold
        self.n_items = len(self.items)
        self._onboarding: list[int] | None = None

    # ------------------------------------------------------------------ core
    def _top(self, s: np.ndarray, k: int) -> pd.DataFrame:
        k = min(k, int(np.isfinite(s).sum()))
        if k <= 0:
            return self.items.iloc[:0].assign(score=[])
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return self.items.iloc[idx].assign(score=s[idx])

    def recommend(self, liked, seen=(), k: int = 10, model: str = "Hybrid",
                  category: str | None = None, only_new: bool = False) -> pd.DataFrame:
        liked = list(liked)
        s = np.asarray(self.models[model].score(np.asarray(liked, dtype=int)), dtype=np.float64)
        blocked = np.zeros(self.n_items, bool)
        blocked[list(set(seen) | set(liked))] = True
        if category and category != "All":
            blocked |= self.items["category"].to_numpy() != category
        if only_new:
            blocked |= ~self.items["is_new"].to_numpy()
        s[blocked] = -np.inf
        return self._top(s, k)

    def similar(self, item: int, model: str = "Content", k: int = 6) -> pd.DataFrame:
        s = np.asarray(self.models[model].score(np.array([item])), dtype=np.float64)
        s[item] = -np.inf
        return self._top(s, k)

    def similar_to_text(self, text: str, k: int = 6) -> pd.DataFrame:
        """Item cold start in its purest form: rank the catalogue for a product that has no
        interactions and is not even in the catalogue, using only its text."""
        c = self.models["Content"]
        return self._top(np.asarray(c.X @ c.encode(text), dtype=np.float64).ravel(), k)

    # ------------------------------------------------------------------ helpers for the UI
    def weights(self, n: int) -> dict:
        return self.models["Hybrid"].weights(n)

    def categories(self, top: int = 25) -> list[str]:
        return self.items["category"].value_counts().head(top).index.tolist()

    def search(self, query: str, limit: int = 15) -> pd.DataFrame:
        m = self.items["title"].str.contains(query, case=False, regex=False)
        return self.items[m].sort_values("n_pos", ascending=False).head(limit)

    def onboarding_items(self, n: int = 16) -> list[int]:
        """Popular items spread over the biggest categories: the new-user questionnaire."""
        if self._onboarding is None:
            cats = self.items.groupby("category")["n_pos"].sum().nlargest(max(4, n // 3)).index
            per = max(1, n // len(cats))
            picks: list[int] = []
            for c in cats:
                picks += self.items[self.items["category"] == c].nlargest(per, "n_pos").index.tolist()
            self._onboarding = picks[:n]
        return self._onboarding

    def random_new_item(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self.items.index[self.items["is_new"]]))

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "RecSys":
        return joblib.load(path)

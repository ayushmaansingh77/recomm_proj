"""Data layer: Amazon Reviews 2023 (McAuley Lab) loader, synthetic fallback, preprocessing."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from config import DATA_DIR, HF_BASE


@dataclass
class Dataset:
    inter: pd.DataFrame   # columns: user, item (int indices), rating, ts (ms)
    items: pd.DataFrame   # row i == item index i
    n_users: int
    n_items: int


# --------------------------------------------------------------------------- download
def _stream_jsonl(url: str, limit: int | None = None):
    # Stream line-by-line to avoid loading hundreds of MB into memory
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        for n, line in enumerate(r.iter_lines()):
            if limit is not None and n >= limit:
                break
            if line:
                yield json.loads(line)


def _first_image(imgs) -> str | None:
    if isinstance(imgs, dict):  # dict-of-lists -> list-of-dicts
        imgs = [dict(zip(imgs.keys(), v)) for v in zip(*imgs.values())]
    for im in imgs or []:
        url = im.get("large") or im.get("hi_res") or im.get("thumb")
        if url:
            return url
    return None


def _clean_meta(m: dict) -> dict:
    cats = [c for c in (m.get("categories") or []) if c]
    category = cats[1] if len(cats) > 1 else (cats[0] if cats else (m.get("main_category") or "Other"))
    try:
        price = float(m.get("price"))
    except (TypeError, ValueError):
        price = None
    return dict(
        asin=m["parent_asin"],
        title=(m.get("title") or "").strip(),
        category=category,
        price=price,
        image=_first_image(m.get("images")),
        store=m.get("store") or "",
        features=" ".join(m.get("features") or [])[:1500],
        description=" ".join(m.get("description") or [])[:1500],
    )


def download_amazon(category: str, max_reviews: int, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the first `max_reviews` reviews + matching item metadata; cache as parquet."""
    DATA_DIR.mkdir(exist_ok=True)
    rp = DATA_DIR / f"{category}_reviews_{max_reviews}.parquet"
    ip = DATA_DIR / f"{category}_items_{max_reviews}.parquet"

    # Return cached files on subsequent runs — skip re-download
    if rp.exists() and ip.exists():
        log(f"Using cached download in {DATA_DIR}")
        return pd.read_parquet(rp), pd.read_parquet(ip)

    log(f"Streaming up to {max_reviews:,} reviews for {category} ...")
    rows = [
        (r["user_id"], r["parent_asin"], float(r["rating"]), int(r["timestamp"]))
        for r in _stream_jsonl(f"{HF_BASE}/review_categories/{category}.jsonl", max_reviews)
    ]
    reviews = pd.DataFrame(rows, columns=["user_id", "asin", "rating", "timestamp"])

    log("Streaming item metadata (this file is large; only matching items are kept) ...")
    wanted = set(reviews["asin"])
    # Filter metadata on the fly — avoids storing the full meta file on disk
    items = [
        _clean_meta(m)
        for m in _stream_jsonl(f"{HF_BASE}/meta_categories/meta_{category}.jsonl")
        if m.get("parent_asin") in wanted
    ]
    items = pd.DataFrame(items)
    reviews.to_parquet(rp)
    items.to_parquet(ip)
    return reviews, items


# --------------------------------------------------------------------------- synthetic
_VOCAB = {
    "Action": "shooter battle combat warrior blaster mission tactical arena stealth soldier explosive mercenary",
    "Racing": "turbo speedway drift circuit rally nitro engine track lap champion garage supercar",
    "Puzzle": "brain logic riddle maze tile match mystery cipher block jigsaw mind twist",
    "Fantasy RPG": "dragon quest sword mage kingdom dungeon elven spell legend guild relic sorcerer",
    "Sports": "football soccer league season goal tournament pitch coach basket stadium trophy slam",
    "Simulation": "farm city builder tycoon craft island village manage cafe zoo planet colony",
}
_GENERIC = "edition deluxe classic pack ultimate collection remastered bundle".split()


def make_synthetic(n_users: int = 4000, n_items: int = 3000, seed: int = 0):
    """Offline stand-in with the same schema as the Amazon download.

    Items launch over time (so some only appear late = genuine cold items), popularity is
    heavy-tailed, and users have 1-2 favourite categories.
    """
    rng = np.random.default_rng(seed)
    cats = list(_VOCAB)
    words = {c: _VOCAB[c].split() for c in cats}
    item_cat = rng.integers(0, len(cats), n_items)
    release = rng.beta(1.2, 1.0, n_items)
    pop = rng.lognormal(0, 1.1, n_items)

    items = pd.DataFrame({
        "asin": [f"B{i:07d}" for i in range(n_items)],
        "title": [
            " ".join(rng.choice(words[cats[c]], 3)).title() + " " + rng.choice(_GENERIC).title()
            for c in item_cat
        ],
        "category": [cats[c] for c in item_cat],
        "price": np.round(rng.uniform(5, 70, n_items), 2),
        "image": None,
        "store": [f"Studio {rng.integers(1, 40)}" for _ in range(n_items)],
        "features": [" ".join(rng.choice(words[cats[c]], 8)) for c in item_cat],
        "description": [" ".join(rng.choice(words[cats[c]], 14)) for c in item_cat],
    })

    t0, span = 1.40e12, 2.0e11
    rows = []
    for u in range(n_users):
        prefs = rng.choice(len(cats), size=int(rng.integers(1, 3)), replace=False)
        n = min(80, 5 + int(rng.pareto(1.6) * 4))
        times = np.sort(rng.uniform(rng.uniform(0, 0.85), 1.0, n))
        w0 = pop * np.where(np.isin(item_cat, prefs), 1.0, 0.04)
        seen: list[int] = []
        for t in times:
            w = w0 * (release <= t)
            w[seen] = 0
            if w.sum() <= 0:
                continue
            i = int(rng.choice(n_items, p=w / w.sum()))
            seen.append(i)
            liked = item_cat[i] in prefs
            r = int(np.clip(round(rng.normal(4.4 if liked else 2.5, 0.8)), 1, 5))
            rows.append((f"U{u:06d}", items.at[i, "asin"], float(r), int(t0 + t * span)))
    reviews = pd.DataFrame(rows, columns=["user_id", "asin", "rating", "timestamp"])
    return reviews, items


# --------------------------------------------------------------------------- preprocessing
def preprocess(reviews: pd.DataFrame, items: pd.DataFrame, min_user: int = 5, min_item: int = 3) -> Dataset:
    reviews = reviews.sort_values("timestamp").drop_duplicates(["user_id", "asin"], keep="last")
    items = items.drop_duplicates("asin")
    items = items[items["title"].str.len() > 0]
    reviews = reviews[reviews["asin"].isin(items["asin"])]

    # 3-pass k-core: removing sparse users can orphan items and vice versa
    for _ in range(3):
        reviews = reviews[reviews["user_id"].map(reviews["user_id"].value_counts()) >= min_user]
        reviews = reviews[reviews["asin"].map(reviews["asin"].value_counts()) >= min_item]

    items = items[items["asin"].isin(reviews["asin"])].reset_index(drop=True)
    item2idx = {a: i for i, a in enumerate(items["asin"])}
    user2idx = {u: i for i, u in enumerate(reviews["user_id"].unique())}

    inter = pd.DataFrame({
        "user": reviews["user_id"].map(user2idx).to_numpy(),
        "item": reviews["asin"].map(item2idx).to_numpy(),
        "rating": reviews["rating"].to_numpy(dtype=float),
        "ts": reviews["timestamp"].to_numpy(dtype=np.int64),
    })

    stats = reviews.groupby("asin")["rating"].agg(["mean", "count"])
    items["average_rating"] = items["asin"].map(stats["mean"])
    items["rating_number"] = items["asin"].map(stats["count"])

    # Title is doubled to give it more weight in TF-IDF relative to free-form descriptions
    items["text"] = (
        (items["title"] + " ") * 2 + items["category"] + " " + items["store"] + " "
        + items["features"] + " " + items["description"]
    )
    return Dataset(inter, items, len(user2idx), len(items))


def load_data(category: str, synthetic: bool = False, max_reviews: int = 400_000, log=print) -> Dataset:
    if synthetic:
        reviews, items = make_synthetic()
    else:
        reviews, items = download_amazon(category, max_reviews, log)
    return preprocess(reviews, items)


def temporal_split(inter: pd.DataFrame, q: float = 0.8):
    """Global temporal split: first q of time -> train, rest -> test (no look-ahead)."""
    cut = inter["ts"].quantile(q)
    return inter[inter["ts"] <= cut], inter[inter["ts"] > cut], cut

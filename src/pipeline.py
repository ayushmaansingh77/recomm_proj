"""End-to-end build: load -> benchmark (held-out users) -> refit on everything -> save artifacts."""
from __future__ import annotations

import json
import time

from config import ARTIFACT_DIR, DEFAULT_CATEGORY, LIKE_THRESHOLD
from src.data import load_data
from src.evaluate import run_benchmark
from src.models import ContentModel, build_models, interaction_matrix
from src.recommender import RecSys


def build_artifacts(category: str = DEFAULT_CATEGORY, synthetic: bool = False, max_reviews: int = 400_000,
                    n_eval_users: int = 1000, embedder: str = "tfidf", log=print):
    t0 = time.time()
    ds = load_data(category, synthetic, max_reviews, log)
    n_pos = int((ds.inter["rating"] >= LIKE_THRESHOLD).sum())
    log(f"Dataset: {ds.n_users:,} users, {ds.n_items:,} items, {len(ds.inter):,} interactions "
        f"({n_pos:,} positive)")

    log("Fitting content model ...")
    content = ContentModel(embedder).fit(ds.items["text"].tolist())

    # Phase 1: benchmark on held-out users (eval users are excluded from model fitting)
    log("Running cold-start benchmark ...")
    results, info = run_benchmark(ds, content, n_eval_users=n_eval_users, log=log)

    # Phase 2: refit on ALL users so the served model is as strong as possible
    log("Refitting on all data for serving ...")
    R = interaction_matrix(ds.inter, ds.n_users, ds.n_items, LIKE_THRESHOLD)
    rs = RecSys(ds.items, build_models(R, content))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    rs.save(ARTIFACT_DIR / "recsys.joblib")
    results.to_csv(ARTIFACT_DIR / "results.csv", index=False)

    # meta.json stores a snapshot of config + dataset stats for the UI header
    meta = dict(category="synthetic demo" if synthetic else category, synthetic=synthetic, embedder=embedder,
                n_users=ds.n_users, n_items=ds.n_items, n_interactions=len(ds.inter),
                sparsity=1 - len(ds.inter) / (ds.n_users * ds.n_items), **info)
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Done in {time.time() - t0:.0f}s -> {ARTIFACT_DIR}")
    return rs, results, meta

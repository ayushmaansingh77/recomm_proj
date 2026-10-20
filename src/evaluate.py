"""Cold-start benchmark.

Protocol (designed to avoid the usual leakage):
  1. Temporal split: interactions before the 80% time quantile are "past", the rest "future".
  2. Evaluation users are held OUT of model fitting entirely, so they really are new users.
  3. For each eval user we reveal only their last k liked past items (k = 0, 1, 3, 5, 10) and
     ask each model to rank the catalogue; the ground truth is what they liked in the future.
  4. Cold ITEMS = items with zero interactions in the fitted data. We report recall restricted
     to those targets (cold_recall), where collaborative models cannot score anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import Dataset, temporal_split
from src.models import ContentModel, Hybrid, build_models, interaction_matrix


def _topk(s: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])]


def run_benchmark(
    ds: Dataset,
    content: ContentModel,
    buckets=(0, 1, 3, 5, 10),
    k: int = 10,
    n_eval_users: int = 1000,
    min_hist: int = 10,
    like: float = 4.0,
    seed: int = 0,
    log=print,
    **hybrid_kw,
):
    rng = np.random.default_rng(seed)
    train, test, cut = temporal_split(ds.inter)
    liked_train = train[train["rating"] >= like]
    liked_test = test[test["rating"] >= like]

    n_hist = liked_train.groupby("user").size()
    eligible = np.intersect1d(n_hist[n_hist >= min_hist].index.to_numpy(), liked_test["user"].unique())
    if len(eligible) == 0:
        raise ValueError("No eligible evaluation users; lower min_hist or load more data.")
    eval_users = rng.choice(eligible, min(n_eval_users, len(eligible)), replace=False)
    eval_set = set(eval_users.tolist())
    log(f"Benchmark: {len(eval_users):,} held-out users, split at {pd.to_datetime(cut, unit='ms'):%Y-%m-%d}")

    fit_inter = train[~train["user"].isin(eval_set)]
    R = interaction_matrix(fit_inter, ds.n_users, ds.n_items, like)
    models = build_models(R, content, **hybrid_kw)
    # Items with zero training interactions — CF has nothing to say about these
    cold_mask = models["Popularity"].counts == 0
    hybrids = {n: m for n, m in models.items() if isinstance(m, Hybrid)}
    base = {n: m for n, m in models.items() if n not in hybrids}

    seen = {u: g["item"].to_numpy() for u, g in train[train["user"].isin(eval_set)].groupby("user")}
    hist = {u: g.sort_values("ts")["item"].to_numpy()
            for u, g in liked_train[liked_train["user"].isin(eval_set)].groupby("user")}
    target = {u: g["item"].to_numpy() for u, g in liked_test[liked_test["user"].isin(eval_set)].groupby("user")}

    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rows, recommended = [], {}
    n_targets = n_cold_targets = 0
    for u in eval_users:
        tgt = target[u]
        ctgt = tgt[cold_mask[tgt]]
        n_targets += len(tgt)
        n_cold_targets += len(ctgt)
        idcg = disc[: min(len(tgt), k)].sum()
        for b in buckets:
            h = hist[u][-b:] if b > 0 else hist[u][:0]
            comp = {n: np.asarray(m.score(h), dtype=np.float64) for n, m in base.items()}
            for hn, hm in hybrids.items():
                # Hybrids reuse the base model scores — avoids scoring twice
                comp[hn] = hm.combine(len(h), comp["Item-kNN"], comp["Content"], comp["Popularity"])
            for name, s in comp.items():
                # Mask already-seen items so the model can't recommend what the user rated before
                s[seen[u]] = -np.inf
                top = _topk(s, k)
                hits = np.isin(top, tgt)
                rows.append(dict(
                    model=name, history=b, user=u,
                    recall=hits.sum() / len(tgt),
                    ndcg=(hits * disc).sum() / idcg,
                    cold_recall=np.isin(top, ctgt).sum() / len(ctgt) if len(ctgt) else np.nan,
                ))
                recommended.setdefault((name, b), set()).update(top.tolist())

    df = (pd.DataFrame(rows).groupby(["model", "history"])
          .agg(recall=("recall", "mean"), ndcg=("ndcg", "mean"), cold_recall=("cold_recall", "mean"))
          .reset_index())
    df["coverage"] = [len(recommended[(m, h)]) / ds.n_items for m, h in zip(df["model"], df["history"])]
    df = df.rename(columns={"recall": f"recall@{k}", "ndcg": f"ndcg@{k}",
                            "cold_recall": f"cold_recall@{k}", "coverage": f"coverage@{k}"})
    info = dict(k=k, n_eval_users=int(len(eval_users)), split_date=str(pd.to_datetime(cut, unit="ms").date()),
                cold_item_share_of_targets=float(n_cold_targets / max(n_targets, 1)),
                n_cold_items=int(cold_mask.sum()))
    return df, info

"""Recommendation models.

Every model exposes `score(hist) -> np.ndarray` over ALL items, where `hist` is an array of the
item indices a user liked. Users are represented only by their history ("fold-in"), so a brand-new
user needs no retraining: 0 items -> cold, a few items -> warming up, many -> warm.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    # Return zeros rather than dividing by zero when all scores are identical
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def interaction_matrix(inter, n_users: int, n_items: int, like: float = 4.0) -> sp.csr_matrix:
    # Binary: 1 if the user rated the item >= like, 0 otherwise
    pos = inter[inter["rating"] >= like]
    return sp.csr_matrix(
        (np.ones(len(pos), dtype=np.float32), (pos["user"].to_numpy(), pos["item"].to_numpy())),
        shape=(n_users, n_items),
    )


# --------------------------------------------------------------------------- baselines
class Popularity:
    name = "Popularity"

    def fit(self, R):
        self.counts = np.asarray(R.sum(0)).ravel()
        self.pop = np.log1p(self.counts)
        return self

    def score(self, hist):
        return self.pop.copy()


class ItemKNN:
    """Item-item cosine CF. Scores sum cos(i, j) over liked i, computed with two sparse
    mat-vecs, so the (huge) item-item matrix is never materialised."""
    name = "Item-kNN"

    def fit(self, R):
        self.n_items = R.shape[1]
        # Column-normalise so each column is a unit vector — cosine sim becomes a dot product
        self.Rn = normalize(R.tocsr().astype(np.float32), axis=0).tocsc()
        self._T = self.Rn.T.tocsr()
        return self

    def score(self, hist):
        hist = np.asarray(hist, dtype=int)
        if hist.size == 0:
            return np.zeros(self.n_items, dtype=np.float32)
        v = np.asarray(self.Rn[:, hist].sum(axis=1)).ravel()
        return self._T @ v

    def __getstate__(self):
        d = self.__dict__.copy()
        # Drop _T on serialisation — it's rebuilt from Rn on load, halving artifact size
        d.pop("_T", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._T = self.Rn.T.tocsr()


class PureSVD:
    """Latent-factor CF. New users are folded in as u = sum of liked item factors."""
    name = "SVD"

    def __init__(self, k: int = 64):
        self.k = k

    def fit(self, R):
        # Clamp k to be within valid SVD range for the given matrix shape
        k = max(2, min(self.k, min(R.shape) - 1))
        svd = TruncatedSVD(k, random_state=0).fit(R)
        self.V = svd.components_.T.astype(np.float32)  # items x k
        self.n_items = R.shape[1]
        return self

    def score(self, hist):
        hist = np.asarray(hist, dtype=int)
        if hist.size == 0:
            return np.zeros(self.n_items, dtype=np.float32)
        # Fold-in: approximate user vector as sum of liked item vectors
        return self.V @ self.V[hist].sum(0)


# --------------------------------------------------------------------------- content
class ContentModel:
    """Item-to-item similarity from text. Works for items with zero interactions."""
    name = "Content"

    def __init__(self, embedder: str = "tfidf", max_features: int = 50_000):
        self.embedder, self.max_features = embedder, max_features

    def fit(self, texts):
        if self.embedder == "sbert":
            from sentence_transformers import SentenceTransformer
            self.X = SentenceTransformer("all-MiniLM-L6-v2").encode(
                list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=True
            ).astype(np.float32)
        else:
            self.vec = TfidfVectorizer(
                max_features=self.max_features, ngram_range=(1, 2), min_df=2,
                sublinear_tf=True, stop_words="english",
            )
            self.X = normalize(self.vec.fit_transform(texts)).tocsr()
        self.n_items = self.X.shape[0]
        return self

    def score(self, hist):
        hist = np.asarray(hist, dtype=int)
        if hist.size == 0:
            return np.zeros(self.n_items, dtype=np.float32)
        profile = np.asarray(self.X[hist].mean(axis=0)).ravel()
        return np.asarray(self.X @ profile).ravel()

    def encode(self, text: str):
        """Embed free text (e.g. a brand-new product that is not in the catalogue yet)."""
        if self.embedder == "sbert":
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2").encode([text], normalize_embeddings=True)[0]
        return np.asarray(normalize(self.vec.transform([text])).todense()).ravel()


# --------------------------------------------------------------------------- hybrid
class Hybrid:
    """Adaptive blend of CF, content and popularity.

    user trust   a = n / (n + k_user)        grows with the user's history length n
    item trust   c = cnt / (cnt + m_item)    grows with the item's interaction count
    CF weight    t = a * c                   (0 for brand-new users OR brand-new items)
    score = t*z(CF) + (1-t) * [ r*z(content) + (1-r)*z(popularity) ]
            + cold_boost * (1-c) * r * max(z(content), 0)          with r = n / (n + 1)

    The last term is an exploration bonus: items the CF side has barely seen get extra credit for
    being content-similar to the user's taste, otherwise they could never outrank observed items.

    n = 0   -> pure popularity (nothing else is available)
    n small -> content-based dominates (it only needs a handful of liked items)
    n large -> CF dominates for well-observed items, while NEW items fall back to content.
    """
    name = "Hybrid"

    def __init__(self, cf, content, pop, k_user: float = 8.0, m_item: float = 5.0, cold_boost: float = 0.0):
        self.cf, self.content, self.pop = cf, content, pop
        self.k_user, self.m_item, self.cold_boost = k_user, m_item, cold_boost
        self.conf = pop.counts / (pop.counts + m_item)

    def weights(self, n: int) -> dict:
        """Component weights for a well-observed item at history length n."""
        a, r = n / (n + self.k_user), n / (n + 1.0)
        return {"Collaborative": a, "Content": (1 - a) * r, "Popularity": (1 - a) * (1 - r)}

    def combine(self, n: int, cf, content, pop) -> np.ndarray:
        a, r = n / (n + self.k_user), n / (n + 1.0)
        t = a * self.conf
        zc = zscore(content)
        base = t * zscore(cf) + (1 - t) * (r * zc + (1 - r) * zscore(pop))
        return base + self.cold_boost * (1 - self.conf) * r * np.clip(zc, 0, None)

    def score(self, hist):
        h = np.asarray(hist, dtype=int)
        return self.combine(len(h), self.cf.score(h), self.content.score(h), self.pop.score(h))


def build_models(R, content: ContentModel, k_user: float = 8.0, m_item: float = 5.0,
                 cold_boost: float = 1.0) -> dict:
    """Two hybrid variants expose the accuracy / new-item-exposure trade-off:
    `Hybrid` ranks for accuracy; `Hybrid (cold-aware)` adds the exploration bonus for new items."""
    pop = Popularity().fit(R)
    knn = ItemKNN().fit(R)
    svd = PureSVD().fit(R)
    return {
        "Popularity": pop, "Item-kNN": knn, "SVD": svd, "Content": content,
        "Hybrid": Hybrid(knn, content, pop, k_user, m_item, 0.0),
        "Hybrid (cold-aware)": Hybrid(knn, content, pop, k_user, m_item, cold_boost),
    }

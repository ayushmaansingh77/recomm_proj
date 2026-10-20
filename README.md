# Cold-Start Recommender on Amazon Reviews 2023

Hybrid recommender (collaborative filtering + content) that handles **new users** and **new items**,
with a Streamlit frontend and a leakage-aware cold-start benchmark.
🔗 Live demo: https://ayushmaansingh77-recomm-proj-app-hn5ayi.streamlit.app/

## Quick start
```bash
pip install -r requirements.txt
python train.py --category Video_Games --max-reviews 400000   # real data (downloads from Hugging Face)
streamlit run app.py
```
No internet / just want to see it? `python train.py --synthetic` (the app also builds this demo by itself
if no artifacts exist). Other categories: `Software`, `Digital_Music`, `All_Beauty`, `Handmade_Products`.
Per category, the review file is streamed and cut at `--max-reviews`; the metadata file is streamed in full
(can be a few hundred MB) and filtered to the items you need. Downloads are cached in `data/`.

## Layout
```
config.py            paths, dataset URL, thresholds
train.py             CLI: download -> benchmark -> refit -> save artifacts/
app.py               Streamlit UI
src/data.py          download, synthetic fallback, k-core filter, temporal split
src/models.py        Popularity, Item-kNN, PureSVD, Content (TF-IDF / SBERT), Hybrid
src/evaluate.py      cold-start benchmark
src/recommender.py   serving wrapper (recommend, similar, search, onboarding)
src/pipeline.py      end-to-end build
```

## Models
Every model scores the whole catalogue from just a user's liked-item list (*fold-in*), so a new user needs no retraining.

| Model | Idea | New user | New item |
|---|---|---|---|
| Popularity | log(1 + #likes) | ✅ | ❌ |
| Item-kNN | sum of cosine(item, liked item) | ❌ | ❌ |
| SVD | PureSVD, user = sum of liked item factors | ❌ | ❌ |
| Content | TF-IDF (or SBERT) over title, category, features, description | ✅ after ≥1 like | ✅ |
| **Hybrid** | adaptive blend (below) | ✅ | partly |
| **Hybrid (cold-aware)** | Hybrid + exploration bonus for new items | ✅ | ✅ |

Hybrid: `a = n/(n+k_user)` (user trust), `c = count/(count+m_item)` (item trust), CF weight `t = a·c`,
`score = t·z(CF) + (1−t)·[r·z(content) + (1−r)·z(pop)]`, `r = n/(n+1)`.
0 likes → pure popularity; few likes → content leads; many likes → CF leads on well-observed items.
The cold-aware variant adds `cold_boost·(1−c)·r·max(z(content),0)`.

## Evaluation protocol
1. Global **temporal split** (80% quantile), no look-ahead.
2. Evaluation users are **held out of training**, so they are truly new (no leakage from their own history).
3. Reveal only each user's last *n* ∈ {0, 1, 3, 5, 10} liked items; rank the full catalogue; ground truth = likes after the split.
4. **Cold items** = zero interactions in the fitted data. `cold_recall` restricts the ground truth to them.
5. Metrics: Recall@10, NDCG@10, catalogue coverage, cold-item recall. "Liked" = rating ≥ 4.

## Findings on the synthetic demo data (use your real run for the report)
- Hybrid had the best overall recall at n=10 (~0.068 vs SVD ~0.051, Item-kNN ~0.047, Popularity ~0.030).
- Every CF model had **0.0 cold-item recall**; only content-based models retrieved new items.
- **Accuracy vs. new-item exposure is a real trade-off.** `Hybrid (cold-aware)` reached ~0.14 cold recall but overall recall fell to ~0.02 (a bonus applied to every low-count item is blunt). Tune `cold_boost`, or reserve a few slots for new items.
- Numbers on real Amazon data will differ; expect much lower absolute recall on a sparse catalogue.

## Deploy on Streamlit Community Cloud
1. Run `python train.py ...` locally and commit `artifacts/` (`recsys.joblib`, `results.csv`, `meta.json`; check each file is <100 MB, lower `--max-reviews` otherwise).
2. Push to GitHub, create the app on share.streamlit.io, main file `app.py`.
3. Without committed artifacts the app builds the synthetic demo on first load.
Keep `embedder=tfidf` for deployment; SBERT pulls in PyTorch and slows cold boots.

## Ideas to extend
Tune `k_user`, `m_item`, `cold_boost` on a validation period; use review text and images for item embeddings;
add a two-tower model; use bandits (LinUCB) for exploration instead of a static bonus; use ratings < 4 as negatives.

"""Tab 3: Under The Hood — Benchmark Metrics & Algorithmic Foundations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def render_under_the_hood_tab(rs, results_df: pd.DataFrame, meta: dict) -> None:
    """Render Tab 3: Mathematical blend curves and offline held-out benchmark results."""
    k_val = meta.get("k", 10)

    st.markdown("### Mathematical blending curve")
    st.caption("As user history $n$ expands, the adaptive hybrid dynamically shifts confidence between Content, Collaborative (Item-kNN), and Global Popularity.")

    ns = np.arange(0, 31)
    weights_matrix = pd.DataFrame([rs.weights(int(n)) for n in ns], index=ns)
    st.line_chart(weights_matrix, x_label="Observed Likes (n)", y_label="Blend Weight", height=250)

    st.markdown("---")
    st.markdown("### Held-out offline benchmark")
    st.caption(
        f"Temporal evaluation performed on {meta.get('n_eval_users', 0):,} genuinely held-out users excluded from training. "
        f"Interactions before {meta.get('split_date', 'N/A')} form the training horizon; subsequent user actions serve as ground truth."
    )

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Cold catalogue items", f"{meta.get('n_cold_items', 0):,}")
    c_m2.metric("Cold targets in holdout", f"{meta.get('cold_item_share_of_targets', 0):.1%}")
    c_m3.metric("Embedder", meta.get("embedder", "tfidf").upper())
    c_m4.metric("Benchmark K", f"Top-{k_val}")

    st.markdown("#### Performance across user history horizon")
    metric_choice = st.selectbox(
        "Evaluation metric",
        options=[f"recall@{k_val}", f"ndcg@{k_val}", f"coverage@{k_val}"],
        index=0
    )

    pivot_chart = results_df.pivot(index="history", columns="model", values=metric_choice)
    st.line_chart(pivot_chart, x_label="Liked items revealed (n)", y_label=metric_choice, height=290)

    st.markdown("---")
    st.markdown(f"#### Cold-item Recall@{k_val} (unseen products in ground truth)")
    st.caption("Target items that had zero training interactions. Models without content fallback score strictly 0.0000.")

    last_history_step = results_df["history"].max()
    cold_recall_comparison = (
        results_df[results_df["history"] == last_history_step]
        .set_index("model")[f"cold_recall@{k_val}"]
        .sort_values(ascending=False)
    )
    st.bar_chart(cold_recall_comparison, color="#5E6AD2", height=230)

    with st.expander("Full benchmark data table"):
        st.dataframe(
            results_df.round(5),
            use_container_width=True,
            hide_index=True
        )

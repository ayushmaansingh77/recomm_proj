"""Amazon RecSys — Cold-Start Recommender Web Application.

Minimal, refined dark UI inspired by Linear, Vercel, and Raycast.
Orchestrates application state, caching, layout tokens, and views.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from config import ARTIFACT_DIR
from src.recommender import RecSys
from theme import apply_theme
from components import render_header, safe_html
from views import (
    render_for_you_tab,
    render_item_cold_start_tab,
    render_under_the_hood_tab,
    render_item_detail_tab
)

st.set_page_config(
    page_title="Amazon RecSys — Cold-Start Recommender",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply minimal dark theme and smooth scrolling
apply_theme()


# -----------------------------------------------------------------------------
# Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained recommendation engine...")
def load_system() -> RecSys:
    path = ARTIFACT_DIR / "recsys.joblib"
    if not path.exists():
        from src.pipeline import build_artifacts
        build_artifacts(synthetic=True, n_eval_users=300, log=lambda *a: None)
    return RecSys.load(path)


@st.cache_data
def load_benchmark():
    res = pd.read_csv(ARTIFACT_DIR / "results.csv")
    meta = json.loads((ARTIFACT_DIR / "meta.json").read_text())
    return res, meta


rs = load_system()
results_df, meta = load_benchmark()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
render_header(
    title="Cold-Start Recommender",
    description="Adaptive hybrid recommendation engine balancing collaborative signals and content embeddings across user and item cold-start regimes.",
    stats=[
        ("Dataset", f"Amazon {meta.get('category', 'Video_Games')}"),
        ("Catalogue", f"{meta.get('n_items', len(rs.items)):,} items"),
        ("Observed Users", f"{meta.get('n_users', 0):,}"),
        ("Interactions", f"{meta.get('n_interactions', 0):,}"),
        ("Matrix Sparsity", f"{meta.get('sparsity', 0.999):.2%}"),
        ("Eval Holdout", f"{meta.get('n_eval_users', 0):,} users"),
    ]
)

# -----------------------------------------------------------------------------
# Session State & Navigation Strip
# -----------------------------------------------------------------------------
st.session_state.setdefault("profile", {})
st.session_state.setdefault("quick_picks", [])
st.session_state.setdefault("inspect_item_idx", 0)
st.session_state.setdefault("active_tab", "For you")

NAV_TABS = ["For you", "Item cold start", "Under the hood", "Item detail"]
if st.session_state["active_tab"] not in NAV_TABS:
    st.session_state["active_tab"] = "For you"

# Dynamic navigation bar that syncs bidirectionally with inspect buttons
safe_html("<div class='nav-tab-container'>")
selected_tab = st.radio(
    "Navigation tabs",
    options=NAV_TABS,
    index=NAV_TABS.index(st.session_state["active_tab"]),
    horizontal=True,
    label_visibility="collapsed"
)
safe_html("</div>")
st.session_state["active_tab"] = selected_tab


# -----------------------------------------------------------------------------
# View Dispatcher
# -----------------------------------------------------------------------------
if selected_tab == "For you":
    render_for_you_tab(rs, results_df, meta)
elif selected_tab == "Item cold start":
    render_item_cold_start_tab(rs, meta)
elif selected_tab == "Under the hood":
    render_under_the_hood_tab(rs, results_df, meta)
elif selected_tab == "Item detail":
    render_item_detail_tab(rs, meta)

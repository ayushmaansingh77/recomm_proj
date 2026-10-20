"""Tab 1: For You — Adaptive Personalized Cold-Start Experience."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import (
    render_regime_banner,
    render_item_list,
    safe_html
)


def format_item_title(rs, item_id: int, max_len: int = 70) -> str:
    """Format catalogue item title with graceful truncation."""
    t = str(rs.items.at[item_id, "title"])
    return t[:max_len] + ("…" if len(t) > max_len else "")


def render_for_you_tab(rs, results_df: pd.DataFrame, meta: dict) -> None:
    """Render Tab 1: Profile building and real-time cold-to-warm recommendations."""
    col_ctrl, col_rec = st.columns([5, 7], gap="large")

    with col_ctrl:
        st.markdown("### Profile builder")
        st.caption("You begin as a cold user with zero history. Each item you like dynamically shifts model weights from content to collaborative filtering.")

        # Onboarding Quick Picks
        onboarding_ids = rs.onboarding_items(16)
        selected_quick = st.multiselect(
            "Quick picks (popular products across major genres)",
            options=onboarding_ids,
            default=st.session_state.quick_picks,
            format_func=lambda idx: format_item_title(rs, idx, 65),
            help="Select known titles to simulate initial user likes"
        )
        st.session_state.quick_picks = selected_quick

        # Search & Rate
        search_query = st.text_input("Search catalogue by title to rate specific games", placeholder="e.g. Zelda, Final Fantasy, Mario...")
        if search_query.strip():
            matches = rs.search(search_query.strip(), limit=8)
            if matches.empty:
                st.caption("No matching titles found.")
            else:
                c_sel, c_star, c_add = st.columns([5, 2, 2])
                with c_sel:
                    selected_match = st.selectbox("Search results", matches.index, format_func=lambda i: format_item_title(rs, i, 55))
                with c_star:
                    rating_score = st.slider("Rating", 1, 5, 5)
                with c_add:
                    st.write("")
                    st.write("")
                    if st.button("Add", key="btn_add_rating", type="primary", use_container_width=True):
                        st.session_state.profile[int(selected_match)] = rating_score
                        st.rerun()

        # Active Profile Summary
        prof = st.session_state.profile
        if prof:
            st.markdown("#### Rated products in session")
            cols_p1, cols_p2 = st.columns([4, 1])
            with cols_p1:
                st.caption(f"{len(prof)} product(s) explicitly rated.")
            with cols_p2:
                if st.button("Clear all", key="btn_clear_profile"):
                    st.session_state.profile = {}
                    st.session_state.quick_picks = []
                    st.rerun()

            for item_id, star in list(prof.items()):
                with st.container(border=True):
                    cp1, cp2, cp3 = st.columns([6, 2, 1])
                    with cp1:
                        safe_html(f"<div style='font-size: 0.875rem; color: #EDEDED; font-weight: 500;'>{format_item_title(rs, item_id, 48)}</div>")
                    with cp2:
                        safe_html(f"<div style='font-size: 0.875rem; color: #5E6AD2;'>{'★' * star}</div>")
                    with cp3:
                        if st.button("✕", key=f"del_{item_id}", help="Remove from profile"):
                            del st.session_state.profile[item_id]
                            st.rerun()

        st.markdown("---")
        st.markdown("### Engine parameters")

        selected_model = st.selectbox(
            "Recommendation model",
            options=["Hybrid (cold-aware)", "Hybrid", "Content", "Item-kNN", "SVD", "Popularity"],
            index=0,
            help="Compare how different algorithmic paradigms respond to the active user profile"
        )

        all_cats = ["All"] + rs.categories(top=20)
        selected_cat = st.selectbox("Filter category", options=all_cats, index=0)

        c_opt1, c_opt2 = st.columns([3, 2])
        with c_opt1:
            only_cold_items = st.toggle("Unseen cold items only (< 5 reviews)", value=False)
        with c_opt2:
            num_recs = st.slider("Results count (K)", min_value=5, max_value=25, value=10, step=5)

    # Compute liked & seen set
    prof = st.session_state.profile
    liked_ids = sorted({i for i, r in prof.items() if r >= 4} | set(st.session_state.quick_picks))
    seen_ids = set(prof.keys()) | set(st.session_state.quick_picks)
    n_liked = len(liked_ids)

    with col_rec:
        st.markdown("### Ranked recommendations")

        # Dynamic blend weights
        weights = rs.weights(n_liked)
        render_regime_banner(n_liked, weights)

        # Blend Weights Chart
        weights_series = pd.Series(weights, name="Weight")
        st.bar_chart(weights_series, height=120, color="#5E6AD2")
        st.caption("Active blend weights for mature catalogue products. Cold items automatically rely on content similarity.")

        # Compute recommendations
        recs = rs.recommend(
            liked=liked_ids,
            seen=seen_ids,
            k=num_recs,
            model=selected_model,
            category=selected_cat if selected_cat != "All" else None,
            only_new=only_cold_items
        )

        render_item_list(
            recs,
            empty_msg="No products match the selected filters. Try switching category to 'All' or toggling off the cold items filter.",
            numbered=True,
            inspectable=True,
            key_prefix="tab1_rec"
        )

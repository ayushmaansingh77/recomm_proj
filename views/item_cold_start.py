"""Tab 2: Item Cold Start — Dual Neighbours & Zero-Shot Simulator."""

from __future__ import annotations

import numpy as np
import streamlit as st

from components import (
    render_item_card,
    render_item_list,
    safe_html
)


def format_item_title(rs, item_id: int, max_len: int = 70) -> str:
    """Format catalogue item title with graceful truncation."""
    t = str(rs.items.at[item_id, "title"])
    return t[:max_len] + ("…" if len(t) > max_len else "")


def render_item_cold_start_tab(rs, meta: dict) -> None:
    """Render Tab 2: Item cold-start dilemma, side-by-side comparison, and text query simulator."""
    st.markdown("### The item cold-start dilemma")
    st.caption(
        "Collaborative filtering relies on overlapping user interaction vectors. When a newly launched game has zero historical interactions, "
        "its dot product with all other items is exactly zero — rendering it completely invisible to pure CF (Item-kNN, SVD). "
        "Content embeddings solve this by projecting title and description text into semantic vector space immediately."
    )

    col_selector, col_detail = st.columns([1, 1], gap="medium")

    with col_selector:
        mode = st.radio("Catalogue inspection mode", ["Inspect new / cold item (< 5 interactions)", "Search catalogue"], horizontal=True)

        if mode == "Inspect new / cold item (< 5 interactions)":
            c_r1, c_r2 = st.columns([4, 1])
            with c_r1:
                st.caption("Select a real item from the catalogue with near-zero interaction history:")
            with c_r2:
                if st.button("Next item", key="btn_next_cold"):
                    rng = np.random.default_rng()
                    st.session_state["chosen_cold_id"] = rs.random_new_item(rng)
                    st.rerun()

            if "chosen_cold_id" not in st.session_state:
                rng = np.random.default_rng()
                st.session_state["chosen_cold_id"] = rs.random_new_item(rng)
            chosen_item = st.session_state["chosen_cold_id"]
        else:
            search_title = st.text_input("Find catalogue product", placeholder="Type title...", key="input_cold_search")
            if search_title.strip():
                match_df = rs.search(search_title.strip(), limit=10)
                if not match_df.empty:
                    chosen_item = int(st.selectbox("Select match", match_df.index, format_func=lambda i: format_item_title(rs, i, 60)))
                else:
                    st.caption("No matching product found.")
                    chosen_item = None
            else:
                chosen_item = 0

    if chosen_item is not None:
        target_row = rs.items.loc[chosen_item]
        with col_detail:
            st.markdown("#### Selected product")
            # Render item card with clean description block enabled
            render_item_card(
                target_row,
                on_inspect=str(chosen_item),
                btn_key=f"target_btn_{chosen_item}",
                show_description=True
            )

        st.markdown("---")
        st.markdown("### Side-by-side nearest neighbour comparison")

        col_content_sim, col_collab_sim = st.columns(2, gap="large")

        with col_content_sim:
            st.markdown("#### Content-based neighbours (TF-IDF Cosine)")
            st.caption("Finds games with overlapping genre descriptions, keywords, and text metadata.")
            content_neighbours = rs.similar(chosen_item, model="Content", k=5)
            render_item_list(content_neighbours, empty_msg="No content neighbours found.", numbered=True, inspectable=True, key_prefix="tab2_content")

        with col_collab_sim:
            st.markdown("#### Collaborative neighbours (Item-kNN Co-occurrence)")
            n_pos = int(target_row.get("n_pos", 0))
            if n_pos == 0:
                safe_html(
                    """
                    <div style='background-color: #141415; border: 1px solid #232326; border-left: 3px solid #8A8A8E; border-radius: 4px; padding: 1.1rem; margin-bottom: 1rem;'>
                        <div style='font-size: 0.95rem; font-weight: 500; color: #EDEDED;'>Zero collaborative co-occurrences</div>
                        <div style='font-size: 0.875rem; color: #9B9BA1; margin-top: 0.35rem; line-height: 1.5;'>
                            This product has 0 recorded training interactions. Pure collaborative filtering cannot establish a single co-occurrence connection, returning no neighbours or arbitrary ties.
                        </div>
                    </div>
                    """
                )
            else:
                st.caption(f"Based on {n_pos:,} shared customer interaction(s).")
            collab_neighbours = rs.similar(chosen_item, model="Item-kNN", k=5)
            render_item_list(collab_neighbours, empty_msg="No collaborative neighbours available.", numbered=True, inspectable=True, key_prefix="tab2_collab")

    st.markdown("---")
    st.markdown("### Zero-shot text query simulator")
    st.caption("Simulate an unreleased product that does not exist in the catalogue. Enter raw text to instantly find the closest catalogue matches via TF-IDF vector projection.")

    raw_text = st.text_area(
        "Product concept description",
        placeholder="e.g. cozy multiplayer farming simulator with pet breeding, cooking mechanics, and seasonal festivals",
        height=95
    )

    if raw_text.strip():
        st.markdown("#### Closest catalogue counterparts")
        synth_neighbours = rs.similar_to_text(raw_text.strip(), k=6)
        render_item_list(synth_neighbours, empty_msg="No similar products found.", numbered=True, inspectable=True, key_prefix="tab2_synth")

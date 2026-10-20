"""Tab 4: Item Detail — In-Depth Catalogue Inspection & Projections."""

from __future__ import annotations

import html
import pandas as pd
import streamlit as st

from components import (
    render_item_list,
    safe_html
)


def format_item_title(rs, item_id: int, max_len: int = 70) -> str:
    """Format catalogue item title with graceful truncation."""
    t = str(rs.items.at[item_id, "title"])
    return t[:max_len] + ("…" if len(t) > max_len else "")


def render_item_detail_tab(rs, meta: dict) -> None:
    """Render Tab 4: Catalogue item inspection with full metadata, tokens, and dual projections."""
    st.markdown("### Catalogue item inspector")
    st.caption("Deep-dive into full item metadata, text tokens, and dual-neighbourhood projections.")

    col_pick, col_spacer = st.columns([6, 6])
    with col_pick:
        detail_search = st.text_input("Search catalogue to inspect another game", placeholder="Type title to search...", key="detail_search_input")
        if detail_search.strip():
            found_df = rs.search(detail_search.strip(), limit=15)
            if not found_df.empty:
                active_detail_id = int(st.selectbox("Search matches", found_df.index, format_func=lambda i: format_item_title(rs, i, 70), key="detail_match_sel"))
            else:
                st.caption("No matching titles found.")
                active_detail_id = st.session_state.get("inspect_item_idx", 0)
        else:
            active_detail_id = st.session_state.get("inspect_item_idx", 0)

    if active_detail_id in rs.items.index:
        d_row = rs.items.loc[active_detail_id]

        with st.container(border=True):
            col_l, col_r = st.columns([2, 5], gap="medium")
            with col_l:
                img = d_row.get("image")
                if isinstance(img, str) and img.strip().startswith("http"):
                    st.image(img, use_container_width=True)
                else:
                    safe_html(
                        """
                        <div style='width: 100%; aspect-ratio: 1; background: #1C1C1E; border: 1px solid #232326; 
                             border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #8A8A8E; font-size: 0.8125rem;'>
                             NO IMAGE AVAILABLE
                        </div>
                        """
                    )

            with col_r:
                is_new = bool(d_row.get("is_new", False))
                badge = "<span class='linear-badge badge-new' style='margin-left: 0.5rem;'>NEW PRODUCT</span>" if is_new else "<span class='linear-badge badge-warm' style='margin-left: 0.5rem;'>MATURE PRODUCT</span>"

                safe_html(f"<h2 style='margin-top: 0; font-size: 1.4rem;'>{html.escape(str(d_row.get('title', 'Untitled')))} {badge}</h2>")

                # Metadata table / pills
                meta_bits = [
                    f"**Category:** {d_row.get('category', 'N/A')}",
                    f"**Store:** {d_row.get('store', 'N/A')}",
                    f"**ASIN:** `{d_row.get('asin', 'N/A')}`",
                ]
                price = d_row.get("price")
                if pd.notna(price) and price is not None:
                    try:
                        meta_bits.append(f"**Price:** ${float(price):.2f}")
                    except (ValueError, TypeError):
                        pass

                rating = d_row.get("average_rating")
                rating_num = d_row.get("rating_number")
                if pd.notna(rating) and rating is not None:
                    meta_bits.append(f"**Rating:** {float(rating):.1f} ★ ({int(rating_num or 0):,} reviews)")

                meta_bits.append(f"**Training Likes:** {int(d_row.get('n_pos', 0)):,}")

                st.markdown(" · ".join(meta_bits))

                st.markdown("#### Description")
                desc_text = str(d_row.get("description", "No description available in catalogue.")).strip()
                if not desc_text or desc_text.lower() == "nan":
                    desc_text = "No description provided in catalogue metadata."

                safe_html(
                    f"""
                    <div style='font-size: 0.9375rem; color: #EDEDED; line-height: 1.65; background: #141415; padding: 0.95rem 1.15rem; border-radius: 6px; border: 1px solid #232326;'>
                        {html.escape(desc_text)}
                    </div>
                    """
                )

        st.markdown("### Related catalogue projections")
        c_proj1, c_proj2 = st.columns(2, gap="large")

        with c_proj1:
            st.markdown("#### Content-based neighbours")
            st.caption("Closest items in TF-IDF vocabulary space.")
            d_content = rs.similar(active_detail_id, model="Content", k=6)
            render_item_list(d_content, empty_msg="No content neighbours found.", numbered=True, inspectable=True, key_prefix="tab4_content")

        with c_proj2:
            st.markdown("#### Collaborative neighbours")
            st.caption("Closest items by user interaction co-occurrence.")
            if int(d_row.get("n_pos", 0)) == 0:
                st.caption("This item has zero recorded training interactions.")
            d_collab = rs.similar(active_detail_id, model="Item-kNN", k=6)
            render_item_list(d_collab, empty_msg="No collaborative neighbours found.", numbered=True, inspectable=True, key_prefix="tab4_collab")

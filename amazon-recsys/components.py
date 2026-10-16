"""Handcrafted UI components for minimal dark theme.

Adheres strictly to:
- Flat surfaces (#141415), hairline borders (#232326)
- Typography: Inter (400, 500)
- Zero gradients, zero neon, zero decorative emojis
- Clean spacing and micro-details
- Safe HTML rendering (no markdown indentation bugs)
"""

from __future__ import annotations

import html
import textwrap
import pandas as pd
import streamlit as st


def safe_html(markup: str) -> None:
    """Render HTML cleanly without markdown 4-space code-block interpretation."""
    cleaned = textwrap.dedent(markup).strip()
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


def render_header(
    title: str,
    description: str,
    stats: list[tuple[str, str]] | None = None
) -> None:
    """Render top page header with typographic title, description, and metadata tags."""
    safe_html(f"<h1>{html.escape(title)}</h1>")
    safe_html(f"<div style='color: #9B9BA1; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0.85rem;'>{html.escape(description)}</div>")

    if stats:
        pills_html = []
        for label, val in stats:
            pills_html.append(
                f"<span class='stat-pill'><span class='stat-dot'></span>"
                f"<span style='color: #9B9BA1;'>{html.escape(label)}:</span> "
                f"<span style='color: #EDEDED; font-weight: 500;'>{html.escape(str(val))}</span></span>"
            )
        pill_row = f"<div style='display: flex; flex-wrap: wrap; gap: 1.25rem; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #232326;'>{''.join(pills_html)}</div>"
        safe_html(pill_row)


def render_regime_banner(n_liked: int, weights: dict[str, float]) -> None:
    """Render cold-to-warm user regime explanation and dynamic blend weights."""
    if n_liked == 0:
        stage = "Cold user state"
        desc = "Zero interaction history. The model falls back strictly to verified catalogue popularity."
    elif n_liked < 5:
        stage = f"Warming user state ({n_liked} items recorded)"
        desc = "Early interaction profile. Content-based semantic matching dominates to bootstrap personalized taste."
    else:
        stage = f"Warm user state ({n_liked} items recorded)"
        desc = "Rich interaction profile. Collaborative filtering takes precedence for mature items, while content signals backstop new products."

    safe_html(
        f"""
        <div class="regime-banner">
            <div class="regime-title">{stage}</div>
            <div class="regime-desc">{desc}</div>
        </div>
        """
    )


def render_item_card(
    row: pd.Series | dict,
    rank: int | None = None,
    on_inspect: str | None = None,
    btn_key: str | None = None,
    show_description: bool = False
) -> None:
    """Render a single product card with clean hairline borders and typography."""
    title_raw = str(row.get("title", "Untitled Product"))
    title_text = title_raw[:110] + ("…" if len(title_raw) > 110 else "")
    category = str(row.get("category", "General"))
    price = row.get("price")
    rating = row.get("average_rating")
    rating_num = row.get("rating_number", 0)
    n_pos = row.get("n_pos", 0)
    is_new = bool(row.get("is_new", False))
    image_url = row.get("image")

    # Format meta pieces
    meta_parts = [category]
    if pd.notna(price) and price is not None:
        try:
            meta_parts.append(f"${float(price):.2f}")
        except (ValueError, TypeError):
            pass

    if pd.notna(rating) and rating is not None and rating > 0:
        try:
            num_str = f"({int(rating_num):,} reviews)" if pd.notna(rating_num) and rating_num else ""
            meta_parts.append(f"{float(rating):.1f} ★ {num_str}")
        except (ValueError, TypeError):
            pass

    meta_parts.append(f"{int(n_pos):,} likes")
    meta_line = " · ".join(meta_parts)

    badge_html = "<span class='linear-badge badge-new'>NEW ITEM</span>" if is_new else ""
    rank_html = f"<span style='color: #9B9BA1; font-size: 0.8125rem; font-weight: 500; margin-right: 0.5rem;'>#{rank}</span>" if rank else ""

    # Optional description block
    desc_html = ""
    if show_description:
        desc_raw = str(row.get("description", "")).strip()
        if desc_raw and desc_raw.lower() != "nan":
            desc_clean = html.escape(desc_raw[:400] + ("…" if len(desc_raw) > 400 else ""))
            desc_html = f"<div style='margin-top: 0.65rem; font-size: 0.875rem; color: #9B9BA1; line-height: 1.6; background: #18181A; border: 1px solid #232326; border-radius: 4px; padding: 0.6rem 0.85rem;'><span style='color: #EDEDED; font-weight: 500;'>Description:</span> {desc_clean}</div>"
        else:
            desc_html = "<div style='margin-top: 0.65rem; font-size: 0.875rem; color: #6E6E73; font-style: italic;'>No description provided in catalogue metadata.</div>"

    with st.container(border=True):
        col_img, col_info, col_act = st.columns([1, 6, 1], gap="small")
        with col_img:
            if isinstance(image_url, str) and image_url.strip().startswith("http"):
                st.image(image_url, use_container_width=True)
            else:
                safe_html(
                    """
                    <div style='width: 100%; aspect-ratio: 1; background: #1C1C1E; border: 1px solid #232326; 
                         border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #8A8A8E; font-size: 0.75rem;'>
                         NO IMAGE
                    </div>
                    """
                )
        with col_info:
            safe_html(
                f"""
                <div style='margin-bottom: 0.3rem;'>
                    {rank_html}
                    <span style='font-size: 1rem; font-weight: 500; color: #EDEDED;'>{html.escape(title_text)}</span>
                    <span style='margin-left: 0.5rem;'>{badge_html}</span>
                </div>
                <div style='font-size: 0.85rem; color: #9B9BA1;'>
                    {html.escape(meta_line)}
                </div>
                {desc_html}
                """
            )
        with col_act:
            if on_inspect is not None:
                bkey = btn_key if btn_key is not None else f"inspect_{on_inspect}"
                if st.button("Inspect", key=bkey, help="View item in detail"):
                    st.session_state["inspect_item_idx"] = int(on_inspect)
                    st.session_state["active_tab"] = "Item detail"
                    st.rerun()



def render_item_list(
    df: pd.DataFrame,
    empty_msg: str = "No items match current criteria.",
    numbered: bool = True,
    inspectable: bool = True,
    key_prefix: str = "item_list",
    show_description: bool = False
) -> None:
    """Render a vertical sequence of cards with guaranteed unique button keys."""
    if df.empty:
        safe_html(
            f"""
            <div style='padding: 2rem 1rem; text-align: center; color: #8A8A8E; font-size: 0.875rem; 
                 border: 1px dashed #232326; border-radius: 6px;'>
                {html.escape(empty_msg)}
            </div>
            """
        )
        return

    for idx, (item_id, row) in enumerate(df.iterrows()):
        rank = (idx + 1) if numbered else None
        inspect_id = str(item_id) if inspectable else None
        b_key = f"btn_{key_prefix}_{item_id}_{idx}"
        render_item_card(row, rank=rank, on_inspect=inspect_id, btn_key=b_key, show_description=show_description)

"""Views package for Amazon RecSys web application."""

from views.for_you import render_for_you_tab
from views.item_cold_start import render_item_cold_start_tab
from views.under_the_hood import render_under_the_hood_tab
from views.item_detail import render_item_detail_tab

__all__ = [
    "render_for_you_tab",
    "render_item_cold_start_tab",
    "render_under_the_hood_tab",
    "render_item_detail_tab",
]

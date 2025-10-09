"""Page modules for the PlainSpend Streamlit application."""

from .overview import render_page as render_overview_page
from .insights import render_page as render_insights_page

__all__ = [
    "render_insights_page",
    "render_overview_page",
]

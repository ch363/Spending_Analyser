"""PlainSpend dashboard with responsive card layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.layout import (
    NAV_LINKS,
    determine_active_page,
    inject_css,
    render_navbar,
    render_sidebar_filters,
    sync_month_query_param,
)
from app.pages import render_insights_page, render_overview_page
from core import AISummaryError, DashboardData, generate_ai_summary, prepare_dashboard_data


st.set_page_config(
    page_title="PlainSpend | Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "seed.csv"



@st.cache_data(show_spinner=False)
def _load_month_options() -> list[str]:
    """Return sorted month keys (YYYY-MM) available in the transaction CSV."""

    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    except FileNotFoundError:
        return []

    if df.empty or "date" not in df.columns:
        return []

    periods = (
        df["date"].dropna().dt.to_period("M").drop_duplicates().sort_values()
    )
    return [str(period) for period in periods]


@st.cache_data(show_spinner=False)
def _load_dashboard_data(month_key: str | None = None) -> DashboardData:
    """Load and cache dashboard data for a given month key from the CSV seed file."""

    target_date = None
    if month_key:
        try:
            target_date = pd.Period(month_key, freq="M").to_timestamp(how="start")
        except (ValueError, TypeError):
            target_date = None

    return prepare_dashboard_data(DATA_PATH, target_date=target_date)


def main() -> None:
    """Application entrypoint for the PlainSpend dashboard."""

    inject_css()
    valid_pages = [link.slug for link in NAV_LINKS if link.enabled]
    active_page = determine_active_page(valid_pages)
    month_options = _load_month_options()
    default_month = month_options[-1] if month_options else None

    raw_month = st.query_params.get("month")
    if isinstance(raw_month, list):
        raw_month = raw_month[0] if raw_month else None

    session_month = st.session_state.get("month_selector")
    selected_month = None
    for candidate in (session_month, raw_month, default_month):
        if candidate and candidate in month_options:
            selected_month = candidate
            break

    render_navbar(active_page, selected_month)

    selected_month = render_sidebar_filters(month_options, active_page, selected_month)
    sync_month_query_param(selected_month)

    data = _load_dashboard_data(selected_month)
    insight_mode = "insights" if active_page == "insights" else "overview"
    ai_insights = _resolve_ai_summary(data, mode=insight_mode)

    if active_page == "insights":
        render_insights_page(data, ai_insights)
    else:
        render_overview_page(data, ai_insights)


def _resolve_ai_summary(data: DashboardData, mode: str = "overview") -> list[str]:
    summary = data["monthly_summary"]
    month_key = summary["month_label"]
    safe_mode = mode if mode in {"overview", "insights"} else "overview"
    cache_key = f"ai_summary::{month_key}::{safe_mode}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    fallback = data.get("insights", [])

    with st.spinner("Generating AI summary…"):
        try:
            insights = generate_ai_summary(data, mode=safe_mode)
        except AISummaryError as exc:
            st.info(f"AI summary unavailable: {exc}")
            insights = fallback
        except Exception:  # pragma: no cover - defensive
            st.warning("AI summary failed. Showing basic insights instead.")
            insights = fallback
        else:
            st.session_state[cache_key] = insights
    st.session_state.setdefault(cache_key, insights)
    return st.session_state[cache_key]


if __name__ == "__main__":
    main()

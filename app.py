"""PlainSpend dashboard with responsive card layout."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from lib.ai_summary import AISummaryError, generate_ai_summary
from lib.summary import (
    DashboardData,
    MonthlySummary,
    ProgressRow,
    VendorRow,
    prepare_dashboard_data,
)
from ui.charts import (
    build_category_chart,
    build_cumulative_chart,
    build_spending_chart,
)
from ui.components import card, inject_css


st.set_page_config(
    page_title="PlainSpend | Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DATA_PATH = Path("data/seed.csv")

NAV_LINKS: list[tuple[str, str, bool]] = [
    ("overview", "Dashboard", True),
    ("insights", "Insights", True),
    ("help", "Help", False),
    ("settings", "Settings", False),
]


def _get_active_page() -> str:
    """Determine the active page from the URL query params or session state."""

    params = st.query_params
    default_page = st.session_state.get("active_page", "overview")
    raw_page = params.get("page", default_page)
    if isinstance(raw_page, list):
        raw_page = raw_page[0]
    valid_pages = {slug for slug, _, is_enabled in NAV_LINKS if is_enabled}
    page = raw_page if raw_page in valid_pages else "overview"

    if st.session_state.get("active_page") != page:
        st.session_state["active_page"] = page

    current_param = params.get("page")
    if isinstance(current_param, list):
        current_param = current_param[0]

    if current_param != page:
        st.query_params["page"] = page

    return page


def _render_navbar(active_page: str) -> None:
    """Render the dashboard navigation bar with active state."""

    link_markup: list[str] = []
    for slug, label, is_enabled in NAV_LINKS:
        css_class = "ps-nav__link"
        aria_current = ""
        if slug == active_page:
            css_class += " is-active"
            aria_current = ' aria-current="page"'

        if is_enabled:
            href = f"?page={slug}"
            link_markup.append(
                f'<a class="{css_class}" href="{href}"{aria_current} data-page="{slug}">{label}</a>'
            )
        else:
            link_markup.append(f'<span class="{css_class} is-disabled">{label}</span>')

    st.markdown(
        f"""
        <nav class="ps-nav">
            <div class="ps-nav__brand">PlainSpend</div>
            <div class="ps-nav__links">{''.join(link_markup)}</div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def _render_this_month_card(summary: MonthlySummary) -> None:
    """Display key metrics for the current month."""

    st.metric("Total spend to date", f"£{summary['total']:,.0f}", summary["delta"])
    metric_cols = st.columns((1, 1, 1))
    metric_cols[0].metric("Avg day", f"£{summary['avg_day']:,.0f}")
    metric_cols[1].metric("Highest day", f"£{summary['highest_day']:,.0f}")
    metric_cols[2].metric("Subscriptions (MTD)", f"£{summary['subscriptions']:,.0f}")
    if summary["days_remaining"] > 0:
        projection_text = (
            f"Projected month-end: £{summary['projection_low']:,.0f}–£{summary['projection_high']:,.0f} "
            f"({int(summary['projection_confidence'] * 100)}% confidence, {summary['days_remaining']} days left)"
        )
    else:
        projection_text = f"Month closed at £{summary['projected_total']:,.0f}"
    st.caption(f"{summary['month_label']} · {projection_text}")
    st.caption(f"Last refreshed {date.today():%d %b %Y}")


def _render_category_card(category_df: pd.DataFrame) -> None:
    """Render the spend-by-category comparison chart and a ranked summary."""

    if category_df.empty:
        st.info("No category spend recorded for this month yet.")
        return

    chart = build_category_chart(category_df)
    st.plotly_chart(chart, use_container_width=True, key="category-donut")

    st.caption("Select a category to inspect details on the right.")


def _render_details_card(
    category_df: pd.DataFrame,
    progress_rows: list[ProgressRow],
    vendor_rows: list[VendorRow],
) -> None:
    """Render detailed category progress trends and vendor breakdown."""

    categories = category_df["Category"].tolist()
    if not categories:
        st.info("No categories available yet.")
        return

    selected_category = st.selectbox("Category", categories, index=0)

    selected_row = category_df.loc[category_df["Category"] == selected_category]
    if not selected_row.empty:
        row = selected_row.iloc[0]
        current_value = float(row["CurrentValue"])
        previous_value = float(row["PreviousValue"])
        change_amount = float(row["ChangeAmount"])
        share_value = float(row["Share"])
        pct_change = row["PctChange"]

        delta_label = f"£{change_amount:,.0f}"
        if pct_change is not None and not pd.isna(pct_change):
            delta_label = f"£{change_amount:,.0f} ({pct_change:+.1%})"

        summary_cols = st.columns((1.2, 1, 1))
        summary_cols[0].metric("Spend this month", f"£{current_value:,.0f}", delta_label)
        summary_cols[1].metric("Share of spend", f"{share_value:.1%}")
        summary_cols[2].metric("Last month", f"£{previous_value:,.0f}")

    filtered_progress = [row for row in progress_rows if row["category"] == selected_category]
    filtered_vendors = [row for row in vendor_rows if row["category"] == selected_category]

    if filtered_progress:
        st.caption("Top merchants this month")
        for row in filtered_progress:
            st.markdown(
                f"<div class='ps-progress-row'><span>{row['label']}</span><span>{row['delta']}</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(row["width"], 0), 100) / 100)
    else:
        st.info("No trend data for this category yet.")

    if filtered_vendors:
        st.caption("Merchant breakdown data available as hover in donut chart.")


def _render_insights_card(insights: list[str]) -> None:
    """Render AI-generated insights as a styled bullet list."""

    items = "".join(f"<li>{item}</li>" for item in insights)
    st.markdown(
        f"<ul class='ps-insights'>{items}</ul>",
        unsafe_allow_html=True,
    )


def _render_runway_card(runway: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    """Show the cash-flow runway gauge with supporting metrics."""

    if not runway:
        st.info("Runway insights are still loading.")
        return

    safe_days = int(runway.get("days_safe", 0))
    days_until_payday = max(int(runway.get("days_until_payday", 0)), 0)
    ratio = 1.0 if days_until_payday == 0 else safe_days / max(days_until_payday, 1)
    ratio = float(min(max(ratio, 0.0), 1.0))

    progress_text = f"{safe_days} safe days · {days_until_payday} until payday"
    st.progress(ratio, text=progress_text)

    meta_cols = st.columns(2)
    meta_cols[0].metric("Next payday", _format_date_label(runway.get("next_payday")))
    meta_cols[1].metric("Burn rate", f"£{runway.get('average_daily_spend', 0):,.0f}/day")

    budget_cols = st.columns(2)
    budget_cols[0].metric("Commitments", f"£{runway.get('upcoming_commitments', 0):,.0f}")
    budget_cols[1].metric("Buffer", f"£{runway.get('remaining_budget', 0):,.0f}")

    st.caption(runway.get("message", ""))

    detail_items = [
        f"Total spend to date · £{summary['total']:,.0f}",
        f"Projection gap · £{runway.get('forecast_gap', 0):,.0f}",
    ]
    st.markdown(
        f"<ul class='ps-insights'>{''.join(f'<li>{item}</li>' for item in detail_items)}</ul>",
        unsafe_allow_html=True,
    )


def _render_recurring_list(entries: Sequence[Mapping[str, Any]]) -> None:
    if not entries:
        st.success("No recurring subscriptions detected yet.")
        return

    items: list[str] = []
    for entry in entries[:6]:
        due_phrase = _format_due_phrase(int(entry.get("days_until_due", 0)))
        next_label = _format_date_label(entry.get("next_date"))
        avg_amount = float(entry.get("average_amount", 0.0))
        interval_label = str(entry.get("interval_label", "Recurring"))
        merchant = entry.get("merchant", "Unknown")
        items.append(
            f"<li><strong>{merchant}</strong> · £{avg_amount:,.2f} {interval_label.lower()}"
            f" · {due_phrase} (next {next_label})</li>"
        )

    st.markdown(
        f"<ul class='ps-insights'>{''.join(items)}</ul>",
        unsafe_allow_html=True,
    )


def _render_duplicate_list(entries: Sequence[Mapping[str, Any]]) -> None:
    if not entries:
        st.success("No duplicate charges spotted this month.")
        return

    items: list[str] = []
    for entry in entries[:5]:
        merchant = entry.get("merchant", "Unknown")
        date_label = _format_date_label(entry.get("date"))
        amounts = entry.get("amounts", [])
        amount_summary = ", ".join(f"£{float(amount):,.2f}" for amount in amounts)
        count = int(entry.get("count", len(amounts)))
        items.append(
            f"<li><strong>{merchant}</strong> · {count} transactions on {date_label}<br /><span style='color:#4B5563'>Amounts: {amount_summary}</span></li>"
        )

    st.markdown(
        f"<ul class='ps-insights'>{''.join(items)}</ul>",
        unsafe_allow_html=True,
    )


def _render_action_checklist(
    cashflow: Mapping[str, Any],
    recurring_entries: Sequence[Mapping[str, Any]],
    duplicate_entries: Sequence[Mapping[str, Any]],
) -> None:
    action_items: list[str] = []

    if cashflow:
        status = "Stay on pace" if cashflow.get("status") == "on_track" else "Tighten spend"
        action_items.append(f"{status}: {cashflow.get('message', '')}")

    if recurring_entries:
        top = recurring_entries[0]
        due_phrase = _format_due_phrase(int(top.get("days_until_due", 0)))
        amount = float(top.get("average_amount", 0.0))
        cadence = str(top.get("interval_label", "Recurring")).lower()
        action_items.append(
            f"Review {top.get('merchant', 'subscription')} (£{amount:,.2f} {cadence}) — {due_phrase}."
        )

    if duplicate_entries:
        dup = duplicate_entries[0]
        count = int(dup.get("count", len(dup.get("amounts", []))))
        action_items.append(
            f"Double-check {dup.get('merchant', 'recent spend')} duplicates on {_format_date_label(dup.get('date'))} ({count} charges)."
        )

    if not action_items:
        st.success("All clear—no immediate follow-ups detected.")
    else:
        st.markdown(
            f"<ul class='ps-insights'>{''.join(f'<li>{item}</li>' for item in action_items)}</ul>",
            unsafe_allow_html=True,
        )


def _format_due_phrase(days: int) -> str:
    if days < 0:
        return f"Overdue by {abs(days)} day{'s' if abs(days) != 1 else ''}"
    if days == 0:
        return "Due today"
    if days == 1:
        return "Due tomorrow"
    return f"Due in {days} days"


def _format_date_label(value: Any) -> str:
    if value is None:
        return "TBC"
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if value is None:
            return "TBC"
    if isinstance(value, pd.Index):
        value = value[0] if not value.empty else None
        if value is None:
            return "TBC"
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(timestamp):
        return "TBC"
    return timestamp.strftime("%d %b")


def _render_dashboard(data: DashboardData, ai_insights: list[str]) -> None:
    """Render the full PlainSpend dashboard layout."""

    summary = data["monthly_summary"]

    row_one_left, row_one_right = st.columns([1, 2], gap="medium")
    with row_one_left:
        with card("This Month", suffix="Normal for you"):
            _render_this_month_card(summary)
    with row_one_right:
        with card("This months daily spend", suffix="Normal for you"):
            chart = build_spending_chart(data["daily_spend_df"])
            st.plotly_chart(chart, use_container_width=True)

    cumulative_df = data["cumulative_spend_df"]
    if summary["days_remaining"] > 0 and not cumulative_df.empty:
        with card("Cumulative projection", suffix="Forecast"):
            cumulative_chart = build_cumulative_chart(cumulative_df)
            st.plotly_chart(cumulative_chart, use_container_width=True)

    row_two_left, row_two_right = st.columns([3, 2], gap="medium")
    with row_two_left:
        with card("Spend by category", suffix="Normal for you"):
            _render_category_card(data["category_df"])
    with row_two_right:
        with card("Details", suffix="Normal for you"):
            _render_details_card(
                data["category_df"],
                data["progress_rows"],
                data["vendor_rows"],
            )

    with card("AI insights", suffix="AI summary"):
        _render_insights_card(ai_insights)


def _render_overview_page(data: DashboardData, ai_insights: list[str]) -> None:
    """Render the main dashboard overview page."""

    st.title("Overview")
    st.caption("Synthetic spending insights for PlainSpend.")
    _render_dashboard(data, ai_insights)


def _render_insights_page(data: DashboardData, ai_insights: list[str]) -> None:
    """Render the insights page with advanced analytics and AI-driven nudges."""

    st.title("Insights")
    st.caption("Deep dives, AI narratives, and areas to watch this month.")

    summary = data["monthly_summary"]
    cashflow = data.get("cashflow_runway", {})
    recurring_entries = data.get("recurring_entries", [])
    duplicate_entries = data.get("duplicate_entries", [])

    highlights_col, runway_col = st.columns((2, 1.2), gap="medium")
    with highlights_col:
        with card("AI highlights", suffix="Generated"):
            if ai_insights:
                _render_insights_card(ai_insights)
            else:
                st.info("AI highlights will appear here once available.")

    with runway_col:
        with card("Cashflow runway", suffix="Payday aware"):
            _render_runway_card(cashflow, summary)

    recurring_col, duplicate_col = st.columns((2, 1), gap="medium")
    with recurring_col:
        with card("Recurring watchlist", suffix="Subscriptions & bills"):
            _render_recurring_list(recurring_entries)

    with duplicate_col:
        with card("Potential duplicates", suffix="Same-day spend"):
            _render_duplicate_list(duplicate_entries)

    with card("Category movers", suffix="Top shifts"):
        category_df = data["category_df"]
        if category_df.empty:
            st.info("Once category spend is available, you'll see the biggest movers here.")
        else:
            movers = (
                category_df[["Category", "CurrentValue", "ChangeAmount", "PctChange"]]
                .sort_values("ChangeAmount", ascending=False)
                .head(5)
                .copy()
            )
            items = []
            for _, row in movers.iterrows():
                pct_change = row["PctChange"]
                pct_display = (
                    f" ({pct_change:+.1%})"
                    if pct_change is not None and not pd.isna(pct_change)
                    else ""
                )
                items.append(
                    f"<li><strong>{row['Category']}</strong> · £{row['CurrentValue']:,.0f}"
                    f" ({row['ChangeAmount']:+,.0f}{pct_display})</li>"
                )
            st.markdown(
                f"<ul class='ps-insights'>{''.join(items)}</ul>",
                unsafe_allow_html=True,
            )

    with card("Action checklist", suffix="Next steps"):
        _render_action_checklist(cashflow, recurring_entries, duplicate_entries)


@st.cache_data(show_spinner=False)
def _load_dashboard_data() -> DashboardData:
    """Load and cache dashboard data from the CSV seed file."""

    return prepare_dashboard_data(DATA_PATH)


def main() -> None:
    """Application entrypoint for the PlainSpend dashboard."""

    inject_css()
    active_page = _get_active_page()
    _render_navbar(active_page)

    data = _load_dashboard_data()
    insight_mode = "insights" if active_page == "insights" else "overview"
    ai_insights = _resolve_ai_summary(data, mode=insight_mode)

    if active_page == "insights":
        _render_insights_page(data, ai_insights)
    else:
        _render_overview_page(data, ai_insights)


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

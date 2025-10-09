"""Overview dashboard page layout."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.layout import card
from core import DashboardData, MonthlySummary, ProgressRow, VendorRow
from visualization import (
    build_category_chart,
    build_cumulative_chart,
    build_spending_chart,
)


def _render_this_month_card(summary: MonthlySummary) -> None:
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
    st.caption(f"Last refreshed {pd.Timestamp.today():%d %b %Y}")


def _render_category_card(category_df: pd.DataFrame) -> None:
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


def _render_dashboard(data: DashboardData, ai_insights: list[str]) -> None:
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
        if ai_insights:
            items = "".join(f"<li>{item}</li>" for item in ai_insights)
            st.markdown(
                f"<ul class='ps-insights'>{items}</ul>",
                unsafe_allow_html=True,
            )
        else:
            st.info("AI insights will appear here once available.")


def render_page(data: DashboardData, ai_insights: list[str]) -> None:
    """Render the overview dashboard page."""

    st.title("Overview")
    st.caption("Synthetic spending insights for PlainSpend.")
    _render_dashboard(data, ai_insights)


__all__ = ["render_page"]

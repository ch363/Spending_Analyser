"""PlainSpend dashboard with responsive card layout."""

from __future__ import annotations

from datetime import date

from typing import TypedDict

import altair as alt
import pandas as pd
import streamlit as st

from ui.components import card, inject_css


st.set_page_config(
    page_title="PlainSpend | Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


class ProgressRow(TypedDict):
    category: str
    label: str
    delta: str
    width: int


class VendorRow(TypedDict):
    category: str
    label: str
    amount: float


class MonthlySummary(TypedDict):
    total: float
    delta: str
    avg_day: float
    highest_day: float
    subscriptions: float


class DashboardData(TypedDict):
    spending_df: pd.DataFrame
    category_df: pd.DataFrame
    progress_rows: list[ProgressRow]
    vendor_rows: list[VendorRow]
    insights: list[str]
    monthly_summary: MonthlySummary


def _build_spending_chart(spending_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(spending_df)
        .mark_area(
            line={"color": "#0C6FFD", "strokeWidth": 3},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(12, 111, 253, 0.45)", offset=0),
                    alt.GradientStop(color="rgba(12, 111, 253, 0.05)", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
        )
        .encode(
            x=alt.X(
                "Day:T",
                axis=alt.Axis(title="", format="%d %b", labelColor="#6B7280"),
            ),
            y=alt.Y(
                "Spending:Q",
                axis=alt.Axis(title="", labelColor="#6B7280"),
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("Day:T", title="Date", format="%d %b"),
                alt.Tooltip("Spending:Q", title="Spending", format="£,.0f"),
            ],
        )
        .properties(height=360)
        .configure_axis(gridColor="#E6EAF2")
    )


def _build_category_chart(category_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(category_df)
        .mark_arc(innerRadius=56, cornerRadius=8)
        .encode(
            theta=alt.Theta("Value", type="quantitative"),
            color=alt.Color(
                "Category",
                type="nominal",
                scale=alt.Scale(
                    domain=category_df["Category"].tolist(),
                    range=["#0C6FFD", "#FF9C41", "#FF595A", "#7A6CFF", "#FFBE3D"],
                ),
                legend=None,
            ),
            tooltip=["Category", alt.Tooltip("Value:Q", title="Value", format="£,.0f")],
        )
        .properties(width=280, height=280)
    )


def _render_navbar() -> None:
        st.markdown(
                """
                <nav class="ps-nav">
                    <div class="ps-nav__brand">PlainSpend</div>
                    <div class="ps-nav__links">
                        <span class="ps-nav__link is-active">Dashboard</span>
                        <span class="ps-nav__link">Insights</span>
                        <span class="ps-nav__link">Help</span>
                        <span class="ps-nav__link">Settings</span>
                    </div>
                </nav>
                """,
                unsafe_allow_html=True,
        )


def _render_this_month_card(summary: MonthlySummary) -> None:
    st.metric("Total spend to date", f"£{summary['total']:,.0f}", summary["delta"])
    metric_cols = st.columns((1, 1, 1))
    metric_cols[0].metric("Avg day", f"£{summary['avg_day']:,.0f}")
    metric_cols[1].metric("Highest day", f"£{summary['highest_day']:,.0f}")
    metric_cols[2].metric("Subscriptions (MTD)", f"£{summary['subscriptions']:,.0f}")
    st.caption(f"Last refreshed {date.today():%d %b %Y}")


def _render_category_card(category_df: pd.DataFrame) -> None:
    chart = _build_category_chart(category_df)
    chart_col, legend_col = st.columns([2, 1])
    with chart_col:
        st.altair_chart(chart, use_container_width=True)
    with legend_col:
        legend_html = "".join(
            f"<div class='ps-legend-item'><span>{row.Category}</span><span>£{row.Value:,}</span></div>"
            for row in category_df.itertuples()
        )
        st.markdown(
            f"<div class='ps-legend'>{legend_html}</div>",
            unsafe_allow_html=True,
        )


def _render_details_card(
    category_df: pd.DataFrame,
    progress_rows: list[ProgressRow],
    vendor_rows: list[VendorRow],
) -> None:
    categories = category_df["Category"].tolist()
    selected_category = st.selectbox("Category", categories, index=0)

    filtered_progress = [row for row in progress_rows if row["category"] == selected_category]
    filtered_vendors = [row for row in vendor_rows if row["category"] == selected_category]

    if filtered_progress:
        for row in filtered_progress:
            st.markdown(
                f"<div class='ps-progress-row'><span>{row['label']}</span><span>{row['delta']}</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(row["width"], 0), 100) / 100)
    else:
        st.info("No trend data for this category yet.")

    if filtered_vendors:
        vendor_df = pd.DataFrame(filtered_vendors)
        vendor_df = vendor_df.sort_values("amount", ascending=False)
        vendor_chart = (
            alt.Chart(vendor_df)
            .mark_bar(color="#0C6FFD", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("amount:Q", title="Spend (£)", axis=alt.Axis(labelColor="#6B7280")),
                y=alt.Y("label:N", sort="-x", title="Merchant", axis=alt.Axis(labelColor="#6B7280")),
                tooltip=[
                    alt.Tooltip("label:N", title="Merchant"),
                    alt.Tooltip("amount:Q", title="Spend", format="£,.0f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(vendor_chart, use_container_width=True)
    else:
        st.warning("No merchant breakdown available.")


def _render_insights_card(insights: list[str]) -> None:
    items = "".join(f"<li>{item}</li>" for item in insights)
    st.markdown(
        f"<ul class='ps-insights'>{items}</ul>",
        unsafe_allow_html=True,
    )


def _render_dashboard(data: DashboardData) -> None:
    summary = data["monthly_summary"]

    row_one_left, row_one_right = st.columns([1, 2], gap="medium")
    with row_one_left:
        with card("This Month", suffix="Normal for you"):
            _render_this_month_card(summary)
    with row_one_right:
        with card("Spending over time", suffix="Normal for you"):
            chart = _build_spending_chart(data["spending_df"])
            st.altair_chart(chart, use_container_width=True)

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

    with card("AI insights", suffix="Normal for you"):
        _render_insights_card(data["insights"])


def _load_static_data() -> DashboardData:
    days = pd.date_range(date(2025, 9, 1), periods=31, freq="D")
    spending = [
        94,
        97,
        101,
        103,
        105,
        110,
        114,
        117,
        120,
        122,
        126,
        129,
        131,
        133,
        136,
        139,
        142,
        145,
        148,
        150,
        154,
        158,
        161,
        165,
        169,
        172,
        176,
        181,
        186,
        191,
        198,
    ]

    category_df = pd.DataFrame(
        {
            "Category": ["Groceries", "Transport", "Eating Out", "Shopping", "Bills"],
            "Value": [420, 180, 260, 340, 310],
        }
    )

    progress_rows: list[ProgressRow] = [
        {"category": "Groceries", "label": "GrocerX", "delta": "+13% vs avg", "width": 88},
        {"category": "Groceries", "label": "Local Market", "delta": "-5% vs avg", "width": 56},
        {"category": "Groceries", "label": "Organic Box", "delta": "+8% vs avg", "width": 68},
        {"category": "Transport", "label": "TfL", "delta": "+4% vs avg", "width": 64},
        {"category": "Transport", "label": "Ride Share", "delta": "+2% vs avg", "width": 52},
        {"category": "Eating Out", "label": "Burger Hub", "delta": "+6% vs avg", "width": 70},
        {"category": "Shopping", "label": "HomeGoods", "delta": "-9% vs avg", "width": 44},
        {"category": "Bills", "label": "Utilities", "delta": "+3% vs avg", "width": 80},
    ]

    vendor_rows: list[VendorRow] = [
        {"category": "Groceries", "label": "GrocerX", "amount": 220},
        {"category": "Groceries", "label": "Local Market", "amount": 110},
        {"category": "Groceries", "label": "Organic Box", "amount": 90},
        {"category": "Transport", "label": "TfL", "amount": 95},
        {"category": "Transport", "label": "Uber", "amount": 68},
        {"category": "Eating Out", "label": "Burger Hub", "amount": 120},
        {"category": "Eating Out", "label": "Sushi Now", "amount": 98},
        {"category": "Shopping", "label": "HomeGoods", "amount": 175},
        {"category": "Bills", "label": "Octopus Energy", "amount": 120},
    ]

    insights = [
        "You've spent <strong>£1,575</strong> so far (+9% vs last). Projection: <strong>£2,190–£2,320</strong>.",
        "Biggest mover: <strong>Groceries +13%</strong> (avg basket +£4 at GrocerX).",
        "Likely duplicate at <strong>CoffeeChain £47</strong> — show refund helper.",
    ]

    monthly_summary: MonthlySummary = {
        "total": 1575,
        "delta": "+9% vs last month",
        "avg_day": 72,
        "highest_day": 132,
        "subscriptions": 95,
    }

    return {
        "spending_df": pd.DataFrame({"Day": days, "Spending": spending}),
        "category_df": category_df,
        "progress_rows": progress_rows,
        "vendor_rows": vendor_rows,
        "insights": insights,
        "monthly_summary": monthly_summary,
    }


def main() -> None:
    inject_css()
    _render_navbar()
    st.title("Overview")
    st.caption("Synthetic spending insights for PlainSpend.")
    data = _load_static_data()
    _render_dashboard(data)


if __name__ == "__main__":
    main()

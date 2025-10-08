"""Streamlit homepage scaffold with Trading 212 inspired styling."""

from __future__ import annotations

from datetime import date

import base64
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TypedDict

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="PlainSpend | Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_styles() -> None:
    primary_blue = "#0C6FFD"
    deep_blue = "#063E9B"
    card_shadow = "0 26px 72px rgba(6, 62, 155, 0.12)"
    card_radius = "28px"

    st.markdown(
        f"""
        <style>
            :root {{
                --primary-blue: {primary_blue};
                --deep-blue: {deep_blue};
                --soft-gray: #f3f7fc;
            }}

            body {{
                background: var(--soft-gray);
            }}

            [data-testid='stAppViewContainer'] > .main {{
                background: var(--soft-gray);
            }}

            [data-testid='stToolbar'],
            [data-testid='stDecoration'] {{
                display: none;
            }}

            [data-testid='stHeader'] {{
                background: transparent;
            }}

            .block-container {{
                padding-top: 3.5rem;
                padding-bottom: 4rem;
                max-width: 1200px;
            }}

            .page-shell-anchor,
            .card-anchor {{
                display: none;
            }}

            /* Outer white shell */
            [data-testid='stVerticalBlock']:has(> .page-shell-anchor) {{
                background: linear-gradient(150deg, rgba(255, 255, 255, 0.98), rgba(240, 245, 255, 0.92));
                box-shadow: 0 32px 96px rgba(6, 62, 155, 0.14);
                border-radius: 36px;
                padding: 3rem;
                gap: 2.5rem;
            }}

            [data-testid='stVerticalBlock']:has(> .page-shell-anchor) > div[data-testid='stVerticalBlock'] {{
                background: transparent;
                padding: 0;
            }}

            /* Card styling driven by anchors */
            [data-testid='stVerticalBlock']:has(> .card-anchor) {{
                background: rgba(255, 255, 255, 0.97);
                border-radius: {card_radius};
                box-shadow: {card_shadow};
                border: 1px solid rgba(12, 111, 253, 0.08);
                padding: 1.65rem 1.75rem;
                margin-bottom: 1.4rem;
                display: flex;
                flex-direction: column;
                gap: 0.95rem;
            }}

            [data-testid='stVerticalBlock']:has(> .card-anchor[data-card-class~="compact"]) {{
                padding: 1.4rem 1.5rem;
                gap: 0.8rem;
            }}

            [data-testid='stVerticalBlock']:has(> .card-anchor[data-card-class~="chart"]) {{
                padding-bottom: 1.6rem;
            }}

            [data-testid='stVerticalBlock']:has(> .card-anchor) .stMetric-value {{
                color: var(--deep-blue);
            }}

            .nav-bar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.35rem 0.6rem;
                border-radius: 20px;
                background: rgba(255, 255, 255, 0.94);
                box-shadow: 0 22px 46px rgba(12, 111, 253, 0.12);
            }}

            .brand {{
                display: flex;
                align-items: center;
                gap: 0.9rem;
                color: var(--deep-blue);
            }}

            .brand-logo {{
                width: 42px;
                height: 42px;
                border-radius: 14px;
                box-shadow: 0 18px 28px rgba(12, 111, 253, 0.2);
            }}

            .brand-initials {{
                width: 44px;
                height: 44px;
                border-radius: 16px;
                background: linear-gradient(140deg, var(--primary-blue), {deep_blue});
                display: inline-flex;
                align-items: center;
                justify-content: center;
                color: #fff;
                font-weight: 700;
                font-size: 1.15rem;
                box-shadow: 0 18px 30px rgba(12, 111, 253, 0.24);
            }}

            .brand-name {{
                font-size: 1.45rem;
                font-weight: 700;
                letter-spacing: 0.01em;
            }}

            .nav-links {{
                display: flex;
                align-items: center;
                gap: 1.9rem;
                font-size: 0.98rem;
                color: #6d7a8f;
            }}

            .nav-link {{
                position: relative;
                font-weight: 500;
            }}

            .nav-link.active {{
                color: var(--deep-blue);
                font-weight: 600;
            }}

            .nav-link.active::after {{
                content: "";
                position: absolute;
                left: 0;
                right: 0;
                bottom: -8px;
                height: 2px;
                background: var(--primary-blue);
                border-radius: 999px;
            }}

            .action-buttons {{
                display: flex;
                gap: 0.75rem;
                align-items: center;
            }}

            .button-pill {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.55rem 1.2rem;
                border-radius: 14px;
                font-size: 0.88rem;
                font-weight: 600;
                border: 1px solid transparent;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}

            .button-pill.secondary {{
                background: rgba(12, 111, 253, 0.08);
                color: var(--primary-blue);
                border-color: rgba(12, 111, 253, 0.12);
            }}

            .button-pill.primary {{
                background: linear-gradient(135deg, var(--primary-blue), {deep_blue});
                color: #fff;
                box-shadow: 0 18px 36px rgba(12, 111, 253, 0.28);
            }}

            .button-pill:hover {{
                transform: translateY(-1px);
                box-shadow: 0 22px 40px rgba(12, 111, 253, 0.32);
            }}

            .card-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
            }}

            .card-meta {{
                display: flex;
                align-items: center;
                gap: 0.6rem;
            }}

            .card-meta:empty {{
                display: none;
            }}

            .card-title {{
                font-size: 1.05rem;
                font-weight: 600;
                color: var(--deep-blue);
            }}

            .status-pill {{
                display: inline-flex;
                align-items: center;
                padding: 0.3rem 0.75rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 600;
                color: var(--primary-blue);
                background: rgba(12, 111, 253, 0.12);
            }}

            .metric-headline {{
                font-size: 2.45rem;
                font-weight: 700;
                color: var(--deep-blue);
            }}

            .metric-delta {{
                font-size: 0.95rem;
                font-weight: 600;
            }}

            .metric-delta.positive {{
                color: #1ba262;
            }}

            .metric-grid {{
                display: grid;
                gap: 0.9rem;
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}

            .metric-item small {{
                display: block;
                color: #8893a8;
                font-weight: 500;
                margin-bottom: 0.35rem;
            }}

            .metric-item span {{
                color: var(--deep-blue);
                font-weight: 600;
            }}

            .legend-item {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-weight: 600;
                color: var(--deep-blue);
                padding: 0.35rem 0;
            }}

            .category-legend {{
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                padding-top: 0.35rem;
            }}

            .legend-item span:last-child {{
                color: var(--deep-blue);
            }}

            .progress-track {{
                height: 8px;
                border-radius: 999px;
                background: #e9eef9;
                overflow: hidden;
            }}

            .progress-bar {{
                height: 100%;
                background: linear-gradient(135deg, rgba(12, 111, 253, 0.9), rgba(12, 111, 253, 0.65));
                border-radius: inherit;
            }}

            .delta-positive {{
                color: #1ba262;
            }}

            .delta-negative {{
                color: #d9534f;
            }}

            .insights-list {{
                display: flex;
                flex-direction: column;
                gap: 0.85rem;
                margin-top: 0.5rem;
                padding-left: 1.1rem;
            }}

            .insight-item {{
                font-size: 0.95rem;
                color: #4e5b72;
            }}

            .insight-item strong {{
                color: var(--deep-blue);
            }}

            .subtle {{
                color: #96a3bb;
                font-size: 0.8rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=1)
def _get_logo_data_uri() -> str:
    logo_path = Path(__file__).resolve().parent / "PlainSpend.png"
    if not logo_path.exists():
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@contextmanager
def _page_shell() -> Iterator[None]:
    st.markdown("<div class='page-shell-anchor'></div>", unsafe_allow_html=True)
    with st.container():
        yield


@contextmanager
def _card(
    name: str,
    title: str,
    *,
    status: str | None = None,
    selector: str | None = None,
    card_class: str | None = None,
) -> Iterator[None]:
    class_tokens = card_class.split() if card_class else []
    attributes = [f"data-card-name=\"{name}\""]
    if class_tokens:
        attributes.append(f"data-card-class=\"{' '.join(class_tokens)}\"")
    attr_string = " " + " ".join(attributes) if attributes else ""

    meta_bits: list[str] = []
    if selector:
        meta_bits.append(f"<span class='selector-pill'>{selector}</span>")
    if status:
        meta_bits.append(f"<span class='status-pill'>{status}</span>")

    meta_html = "".join(meta_bits)
    meta_html = f"<div class='card-meta'>{meta_html}</div>" if meta_html else ""

    st.markdown(f"<div class='card-anchor'{attr_string}></div>", unsafe_allow_html=True)
    with st.container():
        header_html = ["<div class='card-header'>", f"<div class='card-title'>{title}</div>"]
        if meta_html:
            header_html.append(meta_html)
        header_html.append("</div>")
        st.markdown("".join(header_html), unsafe_allow_html=True)
        yield


def _render_navbar() -> None:
    logo_src = _get_logo_data_uri()
    logo_markup = (
        f"<img src='{logo_src}' alt='PlainSpend logo' class='brand-logo' />"
        if logo_src
        else "<span class='brand-initials'>PS</span>"
    )

    st.markdown(
        f"""
        <div class='nav-bar'>
            <div class='brand'>
                {logo_markup}
                <span class='brand-name'>PlainSpend</span>
            </div>
            <div class='nav-links'>
                <span class='nav-link active'>Dashboard</span>
                <span class='nav-link'>Insights</span>
                <span class='nav-link'>Help</span>
                <span class='nav-link'>Settings</span>
            </div>
            <div class='action-buttons'>
                <span class='button-pill secondary'>Upload CSV</span>
                <span class='button-pill primary'>Generate H Summary</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
            x=alt.X("Day:T", axis=alt.Axis(title="", format="%d", labelColor="#96a3bb")),
            y=alt.Y("Spending:Q", axis=alt.Axis(title="", labelColor="#96a3bb"), scale=alt.Scale(domain=[60, 200])),
            tooltip=[
                alt.Tooltip("Day:T", title="Date", format="%d %b"),
                alt.Tooltip("Spending:Q", title="Spending", format="£,.0f"),
            ],
        )
        .properties(height=250)
        .configure_axis(gridColor="#e6ecfb")
    )


def _render_overview_section(spending_df: pd.DataFrame, category_df: pd.DataFrame) -> None:
    refreshed_on = date.today().strftime("%d %b %Y")

    donut_chart = (
        alt.Chart(category_df)
        .mark_arc(innerRadius=56, cornerRadius=10)
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
            tooltip=["Category", "Value"],
        )
        .properties(width=280, height=260)
    )

    left_col, right_col = st.columns([1.05, 1.95], gap="large")
    with left_col:
        with _card("monthly-spend", "Monthly spend", status="Normal for you", card_class="compact summary"):
            st.markdown(
                f"""
                <div class='metric-headline'>£1,575</div>
                <div class='metric-delta positive'>+9% vs last month</div>
                <div class='metric-grid'>
                    <div class='metric-item'>
                        <small>Avg daily</small>
                        <span>£72</span>
                    </div>
                    <div class='metric-item'>
                        <small>Highest day</small>
                        <span>£132</span>
                    </div>
                    <div class='metric-item'>
                        <small>Subscriptions (MTD)</small>
                        <span>£95</span>
                    </div>
                </div>
                <span class='subtle'>Last refreshed {refreshed_on}</span>
                """,
                unsafe_allow_html=True,
            )
        with _card("spend-by-category", "Spend by category", status="Normal for you", card_class="compact category"):
            chart_col, legend_col = st.columns([1.45, 1], gap="large")
            with chart_col:
                st.altair_chart(donut_chart, use_container_width=True)
            with legend_col:
                legend_html = "".join(
                    f"<div class='legend-item'><span>{row.Category}</span><span>£{row.Value}</span></div>"
                    for row in category_df.itertuples()
                )
                st.markdown(
                    f"<div class='category-legend'>{legend_html}</div>",
                    unsafe_allow_html=True,
                )

    with right_col:
        chart = _build_spending_chart(spending_df)
        with _card("spending-over-time", "Spending over time", status="Normal for you", card_class="chart"):
            st.altair_chart(chart, use_container_width=True)


def _render_details_section(progress_rows: list[dict[str, str | int]], vendor_rows: list[dict[str, str]]) -> None:
    detail_cols = st.columns(2, gap="large")

    progress_html = "".join(
        """
        <div class='detail-row'>
            <div class='row-header'>
                <span>{label}</span>
                <span class='{delta_class}'>{delta}</span>
            </div>
            <div class='progress-track'>
                <div class='progress-bar' style='width: {width}%;'></div>
            </div>
        </div>
        """.format(
            label=row["label"],
            delta=row["delta"],
            width=row["width"],
            delta_class="delta-positive" if str(row["delta"]).startswith("+") else "delta-negative",
        )
        for row in progress_rows
    )

    vendor_html = "".join(
        f"<div class='detail-row'><div class='row-header'><span>{row['label']}</span><span>{row['amount']}</span></div></div>"
        for row in vendor_rows
    )

    with detail_cols[0]:
        with _card("category-drilldown", "Category drilldown", selector="Groceries &#9662;", card_class="compact"):
            st.markdown(
                f"""
                <div class='detail-list'>
                    {progress_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with detail_cols[1]:
        with _card("top-merchants", "Top merchants", status="Normal for you", card_class="compact"):
            st.markdown(
                f"""
                <div class='detail-list'>
                    {vendor_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_insights_section(insights: list[str]) -> None:
    insights_html = "".join(f"<li class='insight-item'>{item}</li>" for item in insights)
    with _card("ai-insights", "AI insights", status="Normal for you"):
        st.markdown(
            f"""
            <ul class='insights-list'>
                {insights_html}
            </ul>
            """,
            unsafe_allow_html=True,
        )


class DashboardData(TypedDict):
    spending_df: pd.DataFrame
    category_df: pd.DataFrame
    progress_rows: list[dict[str, str | int]]
    vendor_rows: list[dict[str, str]]
    insights: list[str]


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

    progress_rows = [
        {"label": "GrocerX", "delta": "+13%", "width": 88},
        {"label": "Local Market", "delta": "-5%", "width": 56},
        {"label": "Organic Box", "delta": "+8%", "width": 68},
    ]

    vendor_rows = [
        {"label": "GrocerX", "amount": "£220"},
        {"label": "Local Market", "amount": "£110"},
        {"label": "Organic Box", "amount": "£30"},
    ]

    insights = [
        "You've spent <strong>£1,575</strong> so far (+9% vs last). Projection: <strong>£2,190–£2,320</strong>.",
        "Biggest mover: <strong>Groceries +13%</strong> (avg basket +£4 at GrocerX).",
        "Likely duplicate at <strong>CoffeeChain £47</strong> — show refund helper.",
    ]

    return {
        "spending_df": pd.DataFrame({"Day": days, "Spending": spending}),
        "category_df": category_df,
        "progress_rows": progress_rows,
        "vendor_rows": vendor_rows,
        "insights": insights,
    }


def main() -> None:
    _inject_styles()
    with _page_shell():
        _render_navbar()

        data: DashboardData = _load_static_data()
        _render_overview_section(
            spending_df=data["spending_df"],
            category_df=data["category_df"],
        )
        _render_details_section(
            progress_rows=data["progress_rows"],
            vendor_rows=data["vendor_rows"],
        )
        _render_insights_section(insights=data["insights"])


if __name__ == "__main__":
    main()

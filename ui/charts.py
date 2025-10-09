from __future__ import annotations

from typing import TypeVar

import altair as alt
import pandas as pd
import plotly.express as px

_TIME_FORMAT = "%d %b"
_LABEL_COLOR = "#475569"
_LABEL_FONT = "Inter"
_LABEL_SIZE = 12
_DOMAIN_COLOR = "#CBD5F5"
_GRID_COLOR = "#EEF2FF"


ChartLike = TypeVar("ChartLike", alt.Chart, alt.LayerChart)


def _configure_chart(chart: ChartLike) -> ChartLike:
    """Apply shared axis and view styling to a chart layer."""

    return chart.configure_axis(
        gridColor=_GRID_COLOR,
        domainColor=_DOMAIN_COLOR,
        tickColor=_DOMAIN_COLOR,
        labelColor=_LABEL_COLOR,
        labelFont=_LABEL_FONT,
        labelFontSize=_LABEL_SIZE,
    ).configure_view(strokeWidth=0)


def _time_axis(field: str) -> alt.X:
    """Create a consistently formatted time axis for daily data."""

    return alt.X(
        field,
        axis=alt.Axis(
            title=None,
            format=_TIME_FORMAT,
            labelAngle=0,
            labelPadding=12,
            labelColor=_LABEL_COLOR,
            labelFontSize=_LABEL_SIZE,
            ticks=False,
            domain=True,
            domainColor=_DOMAIN_COLOR,
        ),
    )


def _value_axis(field: str) -> alt.Y:
    """Create a shared y-axis definition without zero locking."""

    return alt.Y(
        field,
        axis=alt.Axis(
            title=None,
            labelColor=_LABEL_COLOR,
            labelFontSize=_LABEL_SIZE,
            labelPadding=8,
            ticks=False,
        ),
        scale=alt.Scale(zero=False),
    )


def build_spending_chart(spending_df: pd.DataFrame) -> alt.LayerChart:
    """Render the daily spend chart with actuals and projection overlays."""

    base = alt.Chart(spending_df).properties(height=280)
    x_enc = _time_axis("Day:T")
    y_enc = _value_axis("Spend:Q")

    actual = base.transform_filter(alt.datum.Series == "Actual")
    projected = base.transform_filter(alt.datum.Series == "Projected")

    area = actual.mark_area(
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(12, 111, 253, 0.55)", offset=0),
                alt.GradientStop(color="rgba(12, 111, 253, 0.08)", offset=1),
            ],
            x1=1,
            x2=1,
            y1=1,
            y2=0,
        ),
        interpolate="monotone",
    ).encode(x=x_enc, y=y_enc)

    actual_line = actual.mark_line(
        color="#0C6FFD",
        strokeWidth=3,
        interpolate="monotone",
    ).encode(
        x=x_enc,
        y=y_enc,
        tooltip=[
            alt.Tooltip("Day:T", title="Date", format=_TIME_FORMAT),
            alt.Tooltip("Spend:Q", title="Actual spend", format="£,.0f"),
        ],
    )

    actual_points = actual.mark_point(
        color="#FF9C41",
        filled=True,
        size=80,
        stroke="#ffffff",
        strokeWidth=2,
    ).encode(x=x_enc, y=y_enc)

    projected_line = projected.mark_line(
        color="#94A3B8",
        strokeDash=[6, 4],
        strokeWidth=2,
        interpolate="monotone",
    ).encode(
        x=x_enc,
        y=y_enc,
        tooltip=[
            alt.Tooltip("Day:T", title="Date", format=_TIME_FORMAT),
            alt.Tooltip("Spend:Q", title="Projected spend", format="£,.0f"),
        ],
    )

    projected_points = projected.mark_point(
        color="#94A3B8",
        size=60,
        stroke="#ffffff",
        strokeWidth=1,
    ).encode(x=x_enc, y=y_enc)

    chart = area + actual_line + actual_points + projected_line + projected_points
    return _configure_chart(chart)


def build_cumulative_chart(cumulative_df: pd.DataFrame) -> alt.LayerChart:
    """Render the cumulative spend projection chart."""

    base = alt.Chart(cumulative_df).properties(height=280)
    x_enc = _time_axis("Day:T")
    y_enc = _value_axis("Total:Q")

    actual = base.transform_filter(alt.datum.Series == "Actual")
    projected = base.transform_filter(alt.datum.Series == "Projected")

    actual_area = actual.mark_area(
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(12, 111, 253, 0.5)", offset=0),
                alt.GradientStop(color="rgba(12, 111, 253, 0.1)", offset=1),
            ],
            x1=1,
            x2=1,
            y1=1,
            y2=0,
        ),
        interpolate="monotone",
    ).encode(x=x_enc, y=y_enc)

    actual_line = actual.mark_line(
        color="#0C6FFD",
        strokeWidth=3,
        interpolate="monotone",
    ).encode(
        x=x_enc,
        y=y_enc,
        tooltip=[
            alt.Tooltip("Day:T", title="Date", format=_TIME_FORMAT),
            alt.Tooltip("Total:Q", title="Actual cumulative", format="£,.0f"),
        ],
    )

    actual_points = actual.mark_point(
        color="#0C6FFD",
        filled=True,
        size=60,
        stroke="#ffffff",
        strokeWidth=1.5,
    ).encode(x=x_enc, y=y_enc)

    projected_line = projected.mark_line(
        color="#94A3B8",
        strokeDash=[6, 4],
        strokeWidth=2,
        interpolate="monotone",
    ).encode(
        x=x_enc,
        y=y_enc,
        tooltip=[
            alt.Tooltip("Day:T", title="Date", format=_TIME_FORMAT),
            alt.Tooltip("Total:Q", title="Projected cumulative", format="£,.0f"),
        ],
    )

    projected_points = projected.mark_point(
        color="#94A3B8",
        size=55,
        stroke="#ffffff",
        strokeWidth=1,
    ).encode(x=x_enc, y=y_enc)

    chart = actual_area + actual_line + actual_points + projected_line + projected_points
    return _configure_chart(chart)


def build_category_chart(category_df: pd.DataFrame):
    """Render a donut chart for category spend distribution using Plotly."""

    palette = [
        "#0C6FFD",
        "#5DA9FF",
        "#FF3B30",
        "#F97316",
        "#22C55E",
        "#7C3AED",
        "#F59E0B",
        "#FACC15",
    ]

    if category_df.empty:
        empty = pd.DataFrame({"Category": [], "CurrentValue": []})
        fig = px.pie(empty, names="Category", values="CurrentValue", hole=0.55)
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    data = category_df.sort_values("CurrentValue", ascending=False).reset_index(drop=True)
    if len(data) > len(palette):
        repeats = (len(data) // len(palette)) + 1
        color_sequence = (palette * repeats)[: len(data)]
    else:
        color_sequence = palette[: len(data)]

    fig = px.pie(
        data,
        names="Category",
        values="CurrentValue",
        hole=0.55,
        color="Category",
        color_discrete_sequence=color_sequence,
    )

    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{percent:.1%}",
        customdata=data[["CurrentValue", "Share", "ChangeAmount", "PctChange"]],
        hovertemplate=(
            "%{label}<br>"
            "Spend: £%{customdata[0]:,.0f}<br>"
            "Share: %{customdata[1]:.1%}<br>"
            "Change: £%{customdata[2]:,.0f}<br>"
            "Change %: %{customdata[3]:+.1%}<extra></extra>"
        ),
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            title="",
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(color=_LABEL_COLOR, family=_LABEL_FONT, size=_LABEL_SIZE),
        ),
        showlegend=True,
    )

    return fig


def build_vendor_chart(vendor_df: pd.DataFrame):
    """Render a horizontal bar chart for vendor spend within a category using Plotly."""

    if vendor_df.empty:
        empty = pd.DataFrame({"label": [], "amount": []})
        fig = px.bar(empty, x="amount", y="label", orientation="h")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    vendor_df = vendor_df.copy()
    vendor_df["formatted_amount"] = vendor_df["amount"].map(lambda x: f"£{x:,.0f}")
    vendor_df["formatted_share"] = vendor_df["share"].map(lambda x: f"{x:.1%}")

    fig = px.bar(
        vendor_df,
        x="amount",
        y="label",
        orientation="h",
        text="formatted_amount",
        color_discrete_sequence=["#0C6FFD"],
    )

    fig.update_traces(
        hovertemplate=(
            "%{y}<br>Spend: %{text}<br>% of category: %{customdata[0]}<extra></extra>"
        ),
        customdata=vendor_df[["formatted_share"]].to_numpy(),
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        margin=dict(l=0, r=10, t=20, b=0),
        xaxis=dict(title="Spend (£)", showgrid=False, zeroline=False),
        yaxis=dict(title="Merchant", automargin=True),
        bargap=0.35,
        height=240,
    )

    return fig


from __future__ import annotations

from typing import TypeVar

import altair as alt
import pandas as pd

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


def build_category_chart(category_df: pd.DataFrame) -> alt.LayerChart:
    """Render a bar-based category comparison chart."""

    if category_df.empty:
        empty = pd.DataFrame({"Category": [], "CurrentValue": []})
        placeholder = alt.Chart(empty).mark_bar().encode(
            x=alt.X("CurrentValue:Q", title="Spend (£)"),
            y=alt.Y("Category:N", title=None),
        )
        return _configure_chart(alt.layer(placeholder))

    data = category_df.copy()
    data = data.sort_values("CurrentValue", ascending=True)
    sort_order = data["Category"].tolist()

    current_peak = float(data["CurrentValue"].max() or 0.0)
    previous_peak = float(data["PreviousValue"].max() or 0.0)
    max_value = max(current_peak, previous_peak)
    if max_value <= 0:
        max_value = 1.0

    x_scale = alt.Scale(domain=(0, max_value * 1.05), nice=True, zero=True)
    y_axis = alt.Y(
        "Category:N",
        sort=sort_order,
        title=None,
        axis=alt.Axis(
            labelColor=_LABEL_COLOR,
            labelFont=_LABEL_FONT,
            labelFontSize=_LABEL_SIZE,
            ticks=False,
        ),
    )

    base = alt.Chart(data).properties(height=320)

    previous = base.mark_bar(color="#E2E8F0", size=22).encode(
        x=alt.X("PreviousValue:Q", scale=x_scale, axis=None),
        y=y_axis,
    )

    rank_domain = sorted(data["Rank"].astype(int).tolist())

    palette = [
        "#3B82F6",
        "#6366F1",
        "#8B5CF6",
        "#D946EF",
        "#F97316",
        "#22C55E",
        "#0EA5E9",
        "#14B8A6",
    ]
    if len(rank_domain) > len(palette):
        repeats = (len(rank_domain) // len(palette)) + 1
        color_range = (palette * repeats)[: len(rank_domain)]
    else:
        color_range = palette[: len(rank_domain)]

    current = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X(
            "CurrentValue:Q",
            scale=x_scale,
            axis=alt.Axis(
                title="Spend (£)",
                labelColor=_LABEL_COLOR,
                labelFont=_LABEL_FONT,
                labelFontSize=_LABEL_SIZE,
                labelPadding=8,
                tickColor=_DOMAIN_COLOR,
                domainColor=_DOMAIN_COLOR,
                gridColor=_GRID_COLOR,
                ticks=False,
            ),
        ),
        y=y_axis,
        color=alt.Color(
            "Rank:O",
            scale=alt.Scale(domain=rank_domain, range=color_range),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Category:N", title="Category"),
            alt.Tooltip("CurrentValue:Q", title="Current spend", format="£,.0f"),
            alt.Tooltip("PreviousValue:Q", title="Last month", format="£,.0f"),
            alt.Tooltip("ChangeAmount:Q", title="Change", format="£,.0f"),
            alt.Tooltip("PctChange:Q", title="Change %", format="+.1%"),
            alt.Tooltip("Share:Q", title="Share", format=".1%"),
        ],
    )

    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        font=_LABEL_FONT,
        fontSize=12,
        color="#1F2937",
    ).encode(
        x=alt.X("CurrentValue:Q", scale=x_scale),
        y=y_axis,
        text=alt.Text("Share:Q", format=".1%"),
    )

    chart = previous + current + labels
    return _configure_chart(chart)


def build_vendor_chart(vendor_df: pd.DataFrame) -> alt.Chart:
    """Render a horizontal bar chart for vendor spend within a category."""

    return (
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


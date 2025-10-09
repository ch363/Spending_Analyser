from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TIME_FORMAT = "%d %b"
_LABEL_COLOR = "#475569"
_LABEL_FONT = "Inter"
_LABEL_SIZE = 12
_DOMAIN_COLOR = "#CBD5F5"
_GRID_COLOR = "#EEF2FF"


def _empty_plotly_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#94A3B8", size=14, family=_LABEL_FONT),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_spending_chart(
    spending_df: pd.DataFrame,
    currency_symbol: str | None = "£",
) -> go.Figure:
    """Render the daily spend chart with a modern Plotly treatment."""

    if spending_df.empty:
        return _empty_plotly_figure("No spend data for this month.")

    df = spending_df.copy()
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Day"])
    df["Series"] = df["Series"].astype(str)

    actual = df[df["Series"].str.lower() == "actual"].copy()
    if actual.empty:
        return _empty_plotly_figure("No spend transactions this month.")

    actual_totals = actual.groupby("Day", as_index=False)["Spend"].sum()
    date_range = pd.date_range(actual_totals["Day"].min(), actual_totals["Day"].max(), freq="D")
    actual_totals = (
        actual_totals.set_index("Day").reindex(date_range, fill_value=0.0).rename_axis("Day").reset_index()
    )
    actual_totals.rename(columns={"Day": "date", "Spend": "value"}, inplace=True)

    currency_prefix = currency_symbol or ""
    hover_template = f"%{{x|%d %b}}<br>{currency_prefix}%{{y:,.2f}}<extra></extra>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_totals["date"],
            y=actual_totals["value"],
            mode="lines+markers",
            name="Daily spend",
            line=dict(color="#2563eb", width=3, shape="spline", smoothing=0.45),
            marker=dict(size=7, color="#2563eb", line=dict(color="#ffffff", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)",
            hovertemplate=hover_template,
        )
    )

    top_points = actual_totals.nlargest(min(3, len(actual_totals)), "value")
    if not top_points.empty:
        fig.add_trace(
            go.Scatter(
                x=top_points["date"],
                y=top_points["value"],
                mode="markers",
                name="Peak days",
                marker=dict(size=10, color="#f97316", line=dict(color="#ffffff", width=2)),
                hovertemplate=hover_template,
                showlegend=False,
            )
        )

    latest_point = actual_totals.iloc[[-1]] if not actual_totals.empty else pd.DataFrame()
    if not latest_point.empty:
        fig.add_trace(
            go.Scatter(
                x=latest_point["date"],
                y=latest_point["value"],
                mode="markers",
                marker=dict(size=10, color="#1d4ed8", symbol="circle", line=dict(color="#ffffff", width=2)),
                hovertemplate=hover_template,
                name="Latest",
                showlegend=False,
            )
        )

    projected = df[df["Series"].str.lower() == "projected"].copy()
    if not projected.empty:
        projected_totals = (
            projected.groupby("Day", as_index=False)
            .agg({"Spend": "sum"})
            .sort_values(by="Day")
        )
        fig.add_trace(
            go.Scatter(
                x=projected_totals["Day"],
                y=projected_totals["Spend"],
                mode="lines+markers",
                name="Projected",
                line=dict(color="#94a3b8", width=2, dash="dash", shape="spline", smoothing=0.3),
                marker=dict(size=6, color="#94a3b8", line=dict(color="#ffffff", width=1.2)),
                hovertemplate=hover_template,
            )
        )

    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Spend",
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)", zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def _compute_ai_projection_path(
    actual_totals: pd.DataFrame,
    projected_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate an expected projection path using a polynomial trend fit."""

    if actual_totals.empty:
        return pd.DataFrame(columns=["Day", "Total"])

    actual_sorted = actual_totals.sort_values("Day").copy()
    start_day = actual_sorted["Day"].iloc[0]
    end_candidates = []
    if not projected_totals.empty:
        end_candidates.append(projected_totals["Day"].max())
    end_candidates.append(actual_sorted["Day"].iloc[-1])
    end_day = max(day for day in end_candidates if pd.notna(day))

    all_days = pd.date_range(start_day, end_day, freq="D")
    if all_days.empty:
        all_days = pd.DatetimeIndex([start_day])

    x = (actual_sorted["Day"] - start_day).dt.days.to_numpy(dtype=float)
    y = actual_sorted["Total"].to_numpy(dtype=float)

    if len(actual_sorted) == 1:
        predictions = np.full(len(all_days), y[0], dtype=float)
    else:
        if len(actual_sorted) >= 6:
            degree = 3
        elif len(actual_sorted) >= 4:
            degree = 2
        else:
            degree = 1
        try:
            coeffs = np.polyfit(x, y, degree)
        except np.linalg.LinAlgError:
            coeffs = np.polyfit(x, y, 1)
        x_all = (all_days - start_day).days.astype(float)
        predictions = np.polyval(coeffs, x_all)
        predictions = np.maximum.accumulate(predictions)
        predictions = np.maximum(predictions, 0.0)

    actual_map = actual_sorted.set_index("Day")["Total"]
    last_total = float(actual_map.iloc[-1])
    expected_values: list[float] = []
    for day, predicted in zip(all_days, predictions):
        if day in actual_map.index:
            value = float(actual_map.loc[day])
            last_total = value
        else:
            value = max(float(predicted), last_total)
            last_total = value
        if not projected_totals.empty:
            projected_match = projected_totals.loc[projected_totals["Day"] == day, "Total"]
            if not projected_match.empty:
                value = min(value, float(projected_match.iloc[-1]))
        expected_values.append(value)

    if not projected_totals.empty:
        max_projected = float(projected_totals["Total"].max())
        expected_values = [min(val, max_projected) for val in expected_values]

    return pd.DataFrame({"Day": all_days, "Total": expected_values})


def build_cumulative_chart(
    cumulative_df: pd.DataFrame,
    currency_symbol: str | None = "£",
) -> go.Figure:
    """Render the cumulative spend projection chart using Plotly and AI trendline."""

    if cumulative_df.empty:
        return _empty_plotly_figure("No cumulative data available.")

    df = cumulative_df.copy()
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Day"])
    df["Series"] = df["Series"].astype(str)

    actual = df[df["Series"].str.lower() == "actual"].copy()
    if actual.empty:
        return _empty_plotly_figure("No actual cumulative spend yet.")

    actual_totals = (
        actual.groupby("Day", as_index=False)
        .agg({"Total": "max"})
        .sort_values(by="Day")
    )
    projected = df[df["Series"].str.lower() == "projected"].copy()
    projected_totals = (
        projected.groupby("Day", as_index=False)
        .agg({"Total": "max"})
        .sort_values(by="Day")
        if not projected.empty
        else pd.DataFrame(columns=["Day", "Total"])
    )

    ai_path = _compute_ai_projection_path(actual_totals, projected_totals)
    ai_future = ai_path[ai_path["Day"] >= actual_totals["Day"].iloc[-1]] if not ai_path.empty else ai_path

    currency_prefix = currency_symbol or ""
    hover_template = f"%{{x|%d %b}}<br>{currency_prefix}%{{y:,.2f}}<extra></extra>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_totals["Day"],
            y=actual_totals["Total"],
            mode="lines",
            name="Actual total",
            line=dict(color="#2563eb", width=3, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.12)",
            hovertemplate=hover_template,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=actual_totals["Day"],
            y=actual_totals["Total"],
            mode="markers",
            marker=dict(size=7, color="#2563eb", line=dict(color="#ffffff", width=1.5)),
            hovertemplate=hover_template,
            name="Actual points",
            showlegend=False,
        )
    )

    if not projected_totals.empty:
        fig.add_trace(
            go.Scatter(
                x=projected_totals["Day"],
                y=projected_totals["Total"],
                mode="lines+markers",
                name="Projection",
                line=dict(color="#94a3b8", width=2, dash="dash", shape="spline", smoothing=0.3),
                marker=dict(size=6, color="#94a3b8", line=dict(color="#ffffff", width=1.2)),
                hovertemplate=hover_template,
            )
        )

    if not ai_future.empty:
        fig.add_trace(
            go.Scatter(
                x=ai_future["Day"],
                y=ai_future["Total"],
                mode="lines",
                name="AI expected",
                line=dict(color="#9333ea", width=2, dash="dot", shape="spline", smoothing=0.3),
                hovertemplate=hover_template,
            )
        )

    latest_point = actual_totals.iloc[[-1]]
    fig.add_trace(
        go.Scatter(
            x=latest_point["Day"],
            y=latest_point["Total"],
            mode="markers",
            marker=dict(size=11, color="#1d4ed8", symbol="circle", line=dict(color="#ffffff", width=2)),
            hovertemplate=hover_template,
            name="Current",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Total spend",
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)", zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


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


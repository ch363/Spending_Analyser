"""Formatting helpers for PlainSpend summaries."""

from __future__ import annotations

from typing import Optional
import pandas as pd

from core.models import ProjectionResult, VendorRow

__all__ = ["build_insights", "format_delta"]


def build_insights(
    *,
    total_spend: float,
    delta_label: str,
    projection: ProjectionResult,
    confidence: float,
    top_category: Optional[pd.Series],
    top_vendor: Optional[VendorRow],
) -> list[str]:
    insights: list[str] = []

    conf_pct = int(round(confidence * 100))
    if abs(projection.high - projection.low) < 1e-6:
        projection_text = f"Projection: <strong>£{projection.projected_total:,.0f}</strong>"
    else:
        projection_text = (
            f"Projection: <strong>£{projection.low:,.0f}–£{projection.high:,.0f}</strong>"
        )

    insights.append(
        (
            f"You've spent <strong>£{total_spend:,.0f}</strong> so far ({delta_label}). "
            f"{projection_text} ({conf_pct}% confidence)."
        )
    )

    if top_category is not None and not top_category.empty:
        insights.append(
            (
                f"Top category: <strong>{top_category['Category']}</strong> at "
                f"£{top_category['CurrentValue']:,.0f}."
            )
        )

    if top_vendor is not None:
        insights.append(
            (
                f"Largest merchant: <strong>{top_vendor['label']}</strong> "
                f"(£{top_vendor['amount']:,.0f})."
            )
        )

    return insights


def format_delta(current: float, previous: float) -> str:
    if previous <= 0:
        if current <= 0:
            return "No change vs last month"
        return "New vs last month"

    change = (current - previous) / previous
    sign = "+" if change >= 0 else ""
    return f"{sign}{change * 100:.1f}% vs last month"

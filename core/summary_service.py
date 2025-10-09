"""Core logic for assembling PlainSpend dashboard summaries."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from analytics.forecast import compute_cashflow_runway
from analytics.recurring import detect_duplicate_transactions, detect_recurring_transactions
from analytics.summary import (
    build_category_breakdown,
    build_daily_and_cumulative_frames,
    build_daily_spend,
    build_progress_rows,
    build_vendor_rows,
    compute_category_total,
    compute_projection,
    prepare_expenses,
    resolve_current_day,
    resolve_target_period,
)
from core.data_loader import load_transactions
from core.formatting import build_insights, format_delta
from core.models import DashboardData, MonthlySummary

__all__ = ["prepare_dashboard_data"]


def prepare_dashboard_data(
    csv_path: str | Path,
    target_date: Optional[date] = None,
    confidence: float = 0.68,
) -> DashboardData:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = load_transactions(csv_path)
    if df.empty:
        raise ValueError("The provided CSV contains no transactions.")

    expenses = prepare_expenses(df)
    if expenses.empty:
        raise ValueError("No spend transactions available after filtering income and duplicates.")

    target_period = resolve_target_period(expenses, target_date)
    previous_period = target_period - 1

    current_month_all = df[df["date"].dt.to_period("M") == target_period]
    current_month_expenses = expenses[expenses["date"].dt.to_period("M") == target_period]
    previous_month_expenses = expenses[expenses["date"].dt.to_period("M") == previous_period]

    month_start = target_period.to_timestamp(how="start")
    month_end = target_period.to_timestamp(how="end")
    current_day = resolve_current_day(current_month_all, month_end)

    daily_spend = build_daily_spend(current_month_expenses, month_start, current_day)
    projection = compute_projection(daily_spend, current_day, month_end, confidence)
    daily_spend_df, cumulative_spend_df = build_daily_and_cumulative_frames(daily_spend, projection)

    total_spend = float(daily_spend.sum())
    prev_total = float(previous_month_expenses["spend"].sum())
    avg_day = float(total_spend / projection.days_elapsed) if projection.days_elapsed else 0.0
    highest_day = float(daily_spend.max()) if not daily_spend.empty else 0.0
    subscriptions_total = compute_category_total(current_month_expenses, "subscriptions")

    delta_label = format_delta(total_spend, prev_total)

    category_df = build_category_breakdown(current_month_expenses, previous_month_expenses)
    progress_rows = build_progress_rows(current_month_expenses, total_spend)
    vendor_rows = build_vendor_rows(current_month_expenses)
    insights = build_insights(
        total_spend=total_spend,
        delta_label=delta_label,
        projection=projection,
        confidence=confidence,
        top_category=category_df.iloc[0] if not category_df.empty else None,
        top_vendor=vendor_rows[0] if vendor_rows else None,
    )

    recurring_entries = detect_recurring_transactions(expenses, current_day)
    duplicate_entries = detect_duplicate_transactions(current_month_expenses)

    incomes = df[df["category"].str.lower() == "income"].copy()
    cashflow_runway = compute_cashflow_runway(
        expenses=expenses,
        incomes=incomes,
        today=current_day,
        total_spend_to_date=total_spend,
        projected_total=projection.projected_total,
        days_remaining=projection.days_remaining,
        recurring_entries=recurring_entries,
    )

    month_label = target_period.strftime("%B %Y")

    summary: MonthlySummary = {
        "total": total_spend,
        "delta": delta_label,
        "avg_day": avg_day,
        "highest_day": highest_day,
        "subscriptions": subscriptions_total,
        "projected_total": projection.projected_total,
        "projection_low": projection.low,
        "projection_high": projection.high,
        "projection_confidence": confidence,
        "days_elapsed": projection.days_elapsed,
        "days_remaining": projection.days_remaining,
        "month_label": month_label,
    }

    return {
        "daily_spend_df": daily_spend_df,
        "cumulative_spend_df": cumulative_spend_df,
        "category_df": category_df,
        "progress_rows": progress_rows,
        "vendor_rows": vendor_rows,
        "insights": insights,
        "monthly_summary": summary,
        "recurring_entries": recurring_entries,
        "duplicate_entries": duplicate_entries,
        "cashflow_runway": cashflow_runway,
    }

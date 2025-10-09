"""Analytics helpers shared across PlainSpend services."""

from analytics.categorisation import (
    build_category_breakdown,
    build_progress_rows,
    build_vendor_rows,
    compute_category_total,
    merchant_display_name,
    merchant_group,
    normalize_merchant,
    prepare_expenses,
)
from analytics.duplicates import DuplicateEntry, detect_duplicate_transactions
from analytics.forecasting import (
    CashflowRunway,
    build_daily_and_cumulative_frames,
    build_daily_spend,
    compute_cashflow_runway,
    compute_projection,
    resolve_current_day,
    resolve_target_period,
)
from analytics.recurring import RecurringEntry, detect_recurring_transactions

__all__ = [
    "merchant_display_name",
    "merchant_group",
    "normalize_merchant",
    "prepare_expenses",
    "compute_category_total",
    "build_category_breakdown",
    "build_progress_rows",
    "build_vendor_rows",
    "DuplicateEntry",
    "detect_duplicate_transactions",
    "CashflowRunway",
    "build_daily_and_cumulative_frames",
    "build_daily_spend",
    "compute_cashflow_runway",
    "compute_projection",
    "resolve_current_day",
    "resolve_target_period",
    "RecurringEntry",
    "detect_recurring_transactions",
]

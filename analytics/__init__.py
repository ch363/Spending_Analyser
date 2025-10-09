"""Analytics helpers shared across PlainSpend services."""

from analytics.categorize import merchant_display_name, merchant_group, normalize_merchant
from analytics.forecast import CashflowRunway, compute_cashflow_runway
from analytics.recurring import (
    DuplicateEntry,
    RecurringEntry,
    detect_duplicate_transactions,
    detect_recurring_transactions,
)
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

__all__ = [
    "merchant_display_name",
    "merchant_group",
    "normalize_merchant",
    "CashflowRunway",
    "compute_cashflow_runway",
    "DuplicateEntry",
    "RecurringEntry",
    "detect_duplicate_transactions",
    "detect_recurring_transactions",
    "build_category_breakdown",
    "build_daily_and_cumulative_frames",
    "build_daily_spend",
    "build_progress_rows",
    "build_vendor_rows",
    "compute_category_total",
    "compute_projection",
    "prepare_expenses",
    "resolve_current_day",
    "resolve_target_period",
]

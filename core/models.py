"""Shared data model definitions for the PlainSpend dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from analytics.duplicates import DuplicateEntry
    from analytics.forecasting import CashflowRunway
    from analytics.recurring import RecurringEntry

class ProgressRow(TypedDict):
    category: str
    label: str
    delta: str
    width: int


class VendorRow(TypedDict):
    category: str
    label: str
    amount: float
    share: float


class MonthlySummary(TypedDict):
    total: float
    delta: str
    avg_day: float
    highest_day: float
    subscriptions: float
    projected_total: float
    projection_low: float
    projection_high: float
    projection_confidence: float
    days_elapsed: int
    days_remaining: int
    month_label: str


class DashboardData(TypedDict):
    daily_spend_df: pd.DataFrame
    cumulative_spend_df: pd.DataFrame
    category_df: pd.DataFrame
    progress_rows: list[ProgressRow]
    vendor_rows: list[VendorRow]
    insights: list[str]
    monthly_summary: MonthlySummary
    recurring_entries: list["RecurringEntry"]
    duplicate_entries: list["DuplicateEntry"]
    cashflow_runway: "CashflowRunway"


@dataclass(frozen=True)
class ProjectionResult:
    projected_total: float
    low: float
    high: float
    days_remaining: int
    days_elapsed: int
    daily_forecast: list["DailyForecastPoint"]


@dataclass(frozen=True)
class DailyForecastPoint:
    date: pd.Timestamp
    mean: float
    std: float


__all__ = [
    "ProgressRow",
    "VendorRow",
    "MonthlySummary",
    "DashboardData",
    "ProjectionResult",
    "DailyForecastPoint",
]

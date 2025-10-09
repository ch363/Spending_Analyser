"""Recurring payment and duplicate transaction detection helpers."""

from __future__ import annotations

from typing import Iterable, TypedDict

import numpy as np
import pandas as pd

from analytics.categorisation import merchant_display_name, merchant_group

__all__ = [
    "RecurringEntry",
    "detect_recurring_transactions",
]


class RecurringEntry(TypedDict):
    """Metadata describing a detected recurring transaction."""

    group_key: str
    merchant: str
    interval_days: int
    interval_label: str
    average_amount: float
    last_amount: float
    last_date: pd.Timestamp
    next_date: pd.Timestamp
    days_until_due: int
    occurrences: int
    confidence: float


_MONTHLY_RANGE = range(28, 32)
_WEEKLY_RANGE = range(6, 9)
_BIWEEKLY_RANGE = range(13, 16)


def detect_recurring_transactions(
    expenses: pd.DataFrame,
    today: pd.Timestamp,
    *,
    amount_tolerance: float = 0.08,
    min_occurrences: int = 2,
) -> list[RecurringEntry]:
    """Identify recurring transactions grouped by normalised merchant name.

    Parameters
    ----------
    expenses:
        DataFrame containing expense transactions with ``description`` and ``spend``.
    today:
        Reference date used for calculating the next expected charge.
    amount_tolerance:
        Relative tolerance used when comparing recurring amounts.
    min_occurrences:
        Minimum number of past charges required to qualify as recurring.

    Returns
    -------
    list[RecurringEntry]
        Sorted list of recurring entries by due date then amount.
    """

    if expenses.empty:
        return []

    spend = expenses.copy()
    spend = spend[spend.get("spend", 0) > 0]
    if spend.empty:
        return []

    spend["group_key"] = spend["description"].map(merchant_group)
    spend["display_name"] = spend["description"].map(merchant_display_name)

    recurring_entries: list[RecurringEntry] = []

    for group_key, group_df in spend.groupby("group_key"):
        group_df = pd.DataFrame(group_df).sort_values(by="date").copy()
        if len(group_df) < min_occurrences:
            continue

        deltas = group_df["date"].diff().dt.days.dropna()
        if deltas.empty:
            continue

        interval_days = _resolve_interval(days=deltas)
        if interval_days is None:
            continue

        amount_series = group_df["spend"].astype(float)
        median_amount = float(amount_series.median())
        if median_amount <= 0:
            continue

        variability = np.abs(amount_series - median_amount) / median_amount
        within_tolerance = variability <= amount_tolerance
        confidence = float(within_tolerance.mean())
        if confidence < 0.6:
            continue

        last_row = group_df.tail(1).iloc[0]
        next_date = last_row["date"] + pd.Timedelta(days=interval_days)
        days_until_due = int((next_date.normalize() - today.normalize()).days)

        recurring_entries.append(
            {
                "group_key": str(group_key),
                "merchant": str(group_df["display_name"].mode().iat[0]),
                "interval_days": int(interval_days),
                "interval_label": _interval_label(interval_days),
                "average_amount": median_amount,
                "last_amount": float(last_row["spend"]),
                "last_date": pd.Timestamp(last_row["date"]),
                "next_date": pd.Timestamp(next_date),
                "days_until_due": days_until_due,
                "occurrences": int(len(group_df)),
                "confidence": confidence,
            }
        )

    recurring_entries.sort(key=lambda row: (row["days_until_due"], -row["average_amount"]))
    return recurring_entries


def _resolve_interval(days: Iterable[float]) -> int | None:
    """Return a canonical interval value if the series fits a supported cadence."""

    values = list(days)
    if len(values) == 0:
        return None

    median_interval = float(np.median(values))
    if np.isnan(median_interval) or median_interval <= 0:
        return None

    rounded = int(round(median_interval))
    if rounded in _MONTHLY_RANGE:
        return 30
    if rounded in _BIWEEKLY_RANGE:
        return 14
    if rounded in _WEEKLY_RANGE:
        return 7
    return None


def _interval_label(interval_days: int) -> str:
    if interval_days >= 28:
        return "Monthly"
    if interval_days >= 13:
        return "Bi-weekly"
    return "Weekly"

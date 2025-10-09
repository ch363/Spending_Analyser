"""Duplicate transaction detection helpers."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

from analytics.categorisation import merchant_display_name, merchant_group

__all__ = [
    "DuplicateEntry",
    "detect_duplicate_transactions",
]


class DuplicateEntry(TypedDict):
    """Represents a cluster of potential duplicate same-day charges."""

    group_key: str
    merchant: str
    date: pd.Timestamp
    count: int
    amounts: list[float]
    total_amount: float


def detect_duplicate_transactions(
    expenses: pd.DataFrame,
    *,
    amount_tolerance: float = 0.5,
) -> list[DuplicateEntry]:
    """Detect potential duplicate charges based on same-day, same-merchant spend."""

    if expenses.empty:
        return []

    spend = expenses.copy()
    spend = spend[spend.get("spend", 0) > 0]
    if spend.empty:
        return []

    spend["group_key"] = spend["description"].map(merchant_group)
    spend["display_name"] = spend["description"].map(merchant_display_name)

    candidates: list[DuplicateEntry] = []
    spend["date_only"] = spend["date"].dt.normalize()

    grouped = spend.groupby(["group_key", "date_only"])
    for (group_key, date_value), group_df in grouped:
        if len(group_df) < 2:
            continue

        amounts = group_df["spend"].to_numpy(dtype=float)
        if not _has_duplicate_like_amounts(amounts, amount_tolerance):
            continue

        candidates.append(
            {
                "group_key": str(group_key),
                "merchant": str(group_df["display_name"].mode().iat[0]),
                "date": pd.Timestamp(date_value),
                "count": int(len(group_df)),
                "amounts": [float(x) for x in amounts.tolist()],
                "total_amount": float(amounts.sum()),
            }
        )

    candidates.sort(key=lambda row: (row["date"], -row["total_amount"]))
    return candidates


def _has_duplicate_like_amounts(amounts: np.ndarray, tolerance: float) -> bool:
    """Return ``True`` when a set of amounts suggests duplicate charges."""

    if len(amounts) < 2:
        return False

    amounts = np.sort(amounts)
    diffs = np.abs(np.diff(amounts))
    if np.any(diffs <= tolerance):
        return True

    denominator = np.maximum(amounts[1:], np.ones_like(amounts[1:]))
    relative = diffs / denominator
    return bool(np.any(relative <= 0.02))

"""Category aggregation and merchant normalisation helpers."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from core.models import ProgressRow, VendorRow

__all__ = [
    "normalize_merchant",
    "merchant_display_name",
    "merchant_group",
    "prepare_expenses",
    "compute_category_total",
    "build_category_breakdown",
    "build_progress_rows",
    "build_vendor_rows",
]

_TRAILING_METADATA = re.compile(
    r"\b(online|ltd|limited|plc|inc|co|uk|gb|help\.uber\.com)\b",
    re.IGNORECASE,
)
_PUNCTUATION = re.compile(r"[^\w\s]")


@lru_cache(maxsize=512)
def normalize_merchant(raw_name: str) -> str:
    """Return a normalised merchant slug suitable for grouping transactions."""

    if not raw_name:
        return "unknown"

    name = raw_name.strip().lower()
    name = _PUNCTUATION.sub(" ", name)
    name = _TRAILING_METADATA.sub("", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


@lru_cache(maxsize=512)
def merchant_display_name(raw_name: str) -> str:
    """Create a clean display label for merchants based on the raw name."""

    if not raw_name:
        return "Unknown merchant"

    cleaned = re.sub(
        r"^(direct debit|standing order|card payment)\s+",
        "",
        raw_name,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip()
    return cleaned.title()


def merchant_group(raw_name: str) -> str:
    """Alias of :func:`normalize_merchant` for semantic clarity."""

    return normalize_merchant(raw_name)


def prepare_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """Return expense transactions enriched with spend values and labels."""

    expenses = df[df["category"].str.lower() != "income"].copy()
    expenses["category_label"] = expenses["category"].str.replace("_", " ").str.title()
    expenses["spend"] = np.where(expenses["amount"] < 0, -expenses["amount"], 0.0)
    refund_mask = expenses["is_refund"] & (expenses["amount"] > 0)
    expenses.loc[refund_mask, "spend"] = -expenses.loc[refund_mask, "amount"]
    return expenses


def compute_category_total(expenses: pd.DataFrame, category_name: str) -> float:
    """Return total spend for ``category_name`` within the supplied expenses."""

    mask = expenses["category"].str.lower() == category_name.lower()
    if not mask.any():
        return 0.0
    total = expenses.loc[mask, "spend"].sum()
    return float(max(total, 0.0))


def build_category_breakdown(
    current_expenses: pd.DataFrame,
    previous_expenses: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame describing top categories and their changes."""

    if current_expenses.empty:
        return pd.DataFrame(
            columns=[
                "Category",
                "CurrentValue",
                "PreviousValue",
                "Share",
                "PctChange",
                "ChangeAmount",
                "Rank",
            ]
        )

    current_totals = current_expenses.groupby("category_label")["spend"].sum().sort_values(ascending=False)
    current_totals = current_totals[current_totals > 0]

    previous_totals = previous_expenses.groupby("category_label")["spend"].sum()

    top_categories = current_totals.head(8)
    total_value = float(current_totals.sum()) if not current_totals.empty else 0.0

    breakdown = top_categories.reset_index().rename(columns={"category_label": "Category", "spend": "CurrentValue"})
    breakdown["PreviousValue"] = breakdown["Category"].map(previous_totals).fillna(0.0)

    current_values = breakdown["CurrentValue"].astype(float)
    prev_values = breakdown["PreviousValue"].astype(float)

    if total_value > 0:
        breakdown["Share"] = current_values / total_value
    else:
        breakdown["Share"] = 0.0

    pct_change = np.where(prev_values > 0, (current_values - prev_values) / prev_values, np.nan)
    breakdown["PctChange"] = pct_change
    breakdown["ChangeAmount"] = current_values - prev_values
    breakdown["Rank"] = np.arange(1, len(breakdown) + 1)

    return breakdown


def build_progress_rows(expenses: pd.DataFrame, total_spend: float) -> list[ProgressRow]:
    """Return summary rows describing top merchants within each category."""

    rows: list[ProgressRow] = []
    if total_spend <= 0:
        return rows

    for category, category_df in expenses.groupby("category_label"):
        category_total = category_df["spend"].sum()
        if category_total <= 0:
            continue

        merchant_totals = category_df.groupby("description")["spend"].sum().sort_values(ascending=False)

        for merchant, amount in merchant_totals.head(3).items():
            if amount <= 0:
                continue
            share = (amount / category_total) * 100
            width = int(np.clip(round((amount / category_total) * 100), 0, 100))
            spend_label = f"£{amount:,.0f}"
            rows.append(
                {
                    "category": str(category),
                    "label": str(merchant_display_name(merchant)),
                    "delta": f"{share:.0f}% of category · {spend_label}",
                    "width": width,
                }
            )

    return rows


def build_vendor_rows(expenses: pd.DataFrame) -> list[VendorRow]:
    """Return per-category top vendor rows including spend share."""

    merchant_totals = expenses.groupby(["category_label", "description"])["spend"].sum().reset_index()
    merchant_totals = merchant_totals[merchant_totals["spend"] > 0]
    if merchant_totals.empty:
        return []

    merchant_totals["category_total"] = merchant_totals.groupby("category_label")["spend"].transform("sum")
    merchant_totals["share"] = merchant_totals["spend"] / merchant_totals["category_total"].replace(0, np.nan)
    merchant_totals = merchant_totals.dropna(subset=["share"])

    merchant_totals = merchant_totals.sort_values("spend", ascending=False)
    top_per_category = merchant_totals.groupby("category_label", group_keys=False).head(5)

    vendor_rows: list[VendorRow] = []
    for record in top_per_category.to_dict(orient="records"):
        vendor_rows.append(
            {
                "category": str(record["category_label"]),
                "label": str(merchant_display_name(record["description"])),
                "amount": float(record["spend"]),
                "share": float(record["share"]),
            }
        )

    return vendor_rows

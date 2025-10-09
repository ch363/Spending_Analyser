"""Data loading utilities for PlainSpend's dashboard pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import pandas as pd

__all__ = ["load_transactions"]


_CACHE_SIZE: Final[int] = 8


@lru_cache(maxsize=_CACHE_SIZE)
def load_transactions(csv_path: str | Path) -> pd.DataFrame:
    """Return a parsed transactions dataframe for the given CSV path.

    Results are cached to avoid redundant disk reads when recomputing
    dashboard data for the same source file multiple times during a session.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df[~df["txn_id"].str.contains("_dup", case=False, na=False)]
    df = df[~df["note"].str.contains("duplicate", case=False, na=False)]
    df["category"] = df["category"].fillna("uncategorized")
    df["is_refund"] = df["is_refund"].fillna(False)
    return df

"""Merchant normalisation helpers used across analytics pipelines."""

from __future__ import annotations

import re
from functools import lru_cache

__all__ = [
    "normalize_merchant",
    "merchant_display_name",
    "merchant_group",
]

_TRAILING_METADATA = re.compile(
    r"\b(online|ltd|limited|plc|inc|co|uk|gb|help\.uber\.com)\b",
    re.IGNORECASE,
)
_PUNCTUATION = re.compile(r"[^\w\s]")


@lru_cache(maxsize=512)
def normalize_merchant(raw_name: str) -> str:
    """Return a normalised merchant slug suitable for grouping transactions.

    Parameters
    ----------
    raw_name:
        Raw transaction description as stored on the CSV.

    Returns
    -------
    str
        Lowercase identifier with punctuation removed, metadata stripped,
        and whitespace collapsed. "unknown" is returned for falsy input.
    """

    if not raw_name:
        return "unknown"

    name = raw_name.strip().lower()
    name = _PUNCTUATION.sub(" ", name)
    name = _TRAILING_METADATA.sub("", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


@lru_cache(maxsize=512)
def merchant_display_name(raw_name: str) -> str:
    """Create a clean display label for merchants based on the raw name.

    Parameters
    ----------
    raw_name:
        Raw transaction description.

    Returns
    -------
    str
        Title-cased description with payment channel prefixes removed.
    """

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

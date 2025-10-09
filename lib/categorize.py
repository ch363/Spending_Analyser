"""Utilities for normalising merchant names and grouping related spend."""

from __future__ import annotations

import re
from functools import lru_cache


_TRAILING_METADATA = re.compile(r"\b(online|ltd|limited|plc|inc|co|uk|gb|help\.uber\.com)\b", re.I)
_PUNCTUATION = re.compile(r"[^\w\s]")


@lru_cache(maxsize=512)
def normalize_merchant(raw_name: str) -> str:
	"""Return a normalised merchant slug suitable for grouping transactions.

	The normalisation removes punctuation, collapses whitespace, and strips
	common company suffixes so that variant descriptions map to the same key.
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
	"""Create a clean display label for merchants based on the raw name."""

	if not raw_name:
		return "Unknown merchant"

	# Remove leading payment channel hints (e.g. "DIRECT DEBIT")
	cleaned = re.sub(r"^(direct debit|standing order|card payment)\s+", "", raw_name, flags=re.I)
	cleaned = cleaned.strip()
	return cleaned.title()


def merchant_group(raw_name: str) -> str:
	"""Alias of :func:`normalize_merchant` for semantic clarity."""

	return normalize_merchant(raw_name)

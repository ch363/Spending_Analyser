"""Shared Plotly theme tokens for PlainSpend visualizations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeTokens:
    time_format: str = "%d %b"
    label_color: str = "#475569"
    label_font: str = "Inter"
    label_size: int = 12
    domain_color: str = "#CBD5F5"
    grid_color: str = "#EEF2FF"
    brand_blue: str = "#2563EB"
    brand_blue_soft: str = "rgba(37, 99, 235, 0.12)"
    brand_blue_focus: str = "#1D4ED8"
    accent_orange: str = "#F97316"
    accent_purple: str = "#9333EA"
    neutral_grey: str = "#94A3B8"
    neutral_white: str = "#FFFFFF"
    neutral_background: str = "rgba(148, 163, 184, 0.25)"
    category_palette: tuple[str, ...] = (
        "#0C6FFD",
        "#5DA9FF",
        "#FF3B30",
        "#F97316",
        "#22C55E",
        "#7C3AED",
        "#F59E0B",
        "#FACC15",
    )
    vendor_bar_color: str = "#0C6FFD"


_TOKENS = ThemeTokens()


def theme_tokens() -> ThemeTokens:
    """Return the shared visualization tokens.

    The tokens are frozen to keep styling consistent between charts and other
    Plotly artefacts.
    """

    return _TOKENS

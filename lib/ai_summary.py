"""AI-assisted summary generation for PlainSpend."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Union

import pandas as pd
import streamlit as st
from openai import APIError, OpenAI

from .summary import DashboardData, ProgressRow

DEFAULT_MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 400


class AISummaryError(RuntimeError):
    """Raised when the AI summary cannot be generated."""


@dataclass(frozen=True)
class AISummaryRequest:
    payload: dict[str, Any]
    month_label: str
    model: str = DEFAULT_MODEL


def _resolve_openai_client() -> OpenAI:
    """Construct an OpenAI client using Streamlit secrets or env vars."""

    api_key: str | None = None
    api_base: str | None = None

    secrets_section: dict[str, Any] | None = None
    try:
        if hasattr(st, "secrets") and "openai" in st.secrets:
            secrets_section = dict(st.secrets["openai"])  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - streamlit secrets access failure
        secrets_section = None

    if secrets_section:
        api_key = secrets_section.get("api_key") or secrets_section.get("OPENAI_API_KEY")
        api_base = secrets_section.get("api_base")

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    api_base = api_base or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise AISummaryError(
            "Missing OpenAI API key. Add it to .streamlit/secrets.toml under [openai]."
        )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base

    return OpenAI(**client_kwargs)


def _build_category_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    cleaned: list[dict[str, Any]] = []
    for row in records:
        cleaned.append(
            {
                "category": str(row.get("Category", "")),
                "current": float(row.get("CurrentValue", 0.0) or 0.0),
                "previous": float(row.get("PreviousValue", 0.0) or 0.0),
                "share": float(row.get("Share", 0.0) or 0.0),
                "pct_change": None
                if pd.isna(row.get("PctChange"))
                else float(row.get("PctChange") or 0.0),
                "change_amount": float(row.get("ChangeAmount", 0.0) or 0.0),
                "rank": int(row.get("Rank", 0)),
            }
        )
    return cleaned


def _select_top(items: Iterable[dict[str, Any]], key: str, count: int, descending: bool) -> list[dict[str, Any]]:
    rows = list(items)
    rows.sort(key=lambda record: record.get(key, 0.0) or 0.0, reverse=descending)
    trimmed: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(key)
        if value in (None, 0.0):
            continue
        if descending and (value or 0.0) > 0:
            trimmed.append(row)
        elif not descending and (value or 0.0) < 0:
            trimmed.append(row)
        if len(trimmed) >= count:
            break
    return trimmed


def _detect_anomalies(categories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for row in categories:
        pct_change = row.get("pct_change")
        share = row.get("share", 0.0) or 0.0
        is_large_shift = pct_change is not None and abs(pct_change) >= 0.5
        is_significant_share = share >= 0.25 and (row.get("change_amount") or 0.0) > 0
        if is_large_shift or is_significant_share:
            anomalies.append(row)
    return anomalies[:5]


ProgressInput = Union[ProgressRow, dict[str, Any]]


def _extract_subscriptions(progress_rows: Iterable[ProgressInput]) -> list[dict[str, Any]]:
    subs: list[dict[str, Any]] = []
    for row in progress_rows:
        if isinstance(row, dict):
            category = str(row.get("category", "")).lower()
            label = str(row.get("label", ""))
            detail = str(row.get("delta", ""))
        else:
            category = str(row.get("category", "")).lower()  # type: ignore[index]
            label = str(row.get("label", ""))  # type: ignore[index]
            detail = str(row.get("delta", ""))  # type: ignore[index]

        if category != "subscriptions":
            continue
        subs.append(
            {
                "merchant": label,
                "detail": detail,
            }
        )
    return subs[:5]


def build_ai_summary_request(data: DashboardData) -> AISummaryRequest:
    """Create a payload ready to be sent to the OpenAI API."""

    summary = data["monthly_summary"]
    categories = _build_category_records(data["category_df"])

    risers = _select_top(categories, "change_amount", count=3, descending=True)
    fallers = _select_top(categories, "change_amount", count=3, descending=False)
    anomalies = _detect_anomalies(categories)
    subs = _extract_subscriptions(data.get("progress_rows", []))

    payload = {
        "month": summary["month_label"],
        "spend_totals": {
            "current_total": summary["total"],
            "month_delta": summary["delta"],
            "avg_day": summary["avg_day"],
            "highest_day": summary["highest_day"],
            "projection": {
                "low": summary["projection_low"],
                "high": summary["projection_high"],
                "confidence": summary["projection_confidence"],
                "days_remaining": summary["days_remaining"],
            },
        },
        "category_changes": {
            "top_risers": risers,
            "top_fallers": fallers,
            "all": categories,
        },
        "anomalies": anomalies,
        "subscriptions": subs,
    }

    model = DEFAULT_MODEL
    try:
        if hasattr(st, "secrets") and "openai" in st.secrets:
            openai_section = st.secrets["openai"]
            if "model" in openai_section:
                model = str(openai_section["model"])
    except Exception:  # pragma: no cover - streamlit secrets access failure
        model = DEFAULT_MODEL

    return AISummaryRequest(payload=payload, month_label=summary["month_label"], model=model)


def generate_ai_summary(data: DashboardData) -> list[str]:
    """Generate a list of bullet point insights using OpenAI."""

    request = build_ai_summary_request(data)
    client = _resolve_openai_client()

    system_prompt = (
        "You are an assistant that writes concise financial summaries for personal spending. "
        "You receive structured JSON summarising the current and previous month's card spending. "
        "Write 3-4 bullet points comparing current vs last month, highlighting: (1) major spend changes, "
        "(2) categories with notable rises/falls or anomalies, and (3) subscriptions to review. "
        "Use professional but friendly tone, UK currency formatting with the pound symbol, and keep it under 120 words."
    )

    formatted_payload = json.dumps(request.payload, ensure_ascii=False, indent=2)
    user_message = (
        "Analyse the monthly spending context below and craft the requested summary.\n\n"
        f"Month: {request.month_label}\n"
        "Data (JSON):\n"
        f"{formatted_payload}"
    )

    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,
        )
    except APIError as exc:
        raise AISummaryError(f"OpenAI API error: {exc}") from exc

    try:
        text = response.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:  # pragma: no cover - unexpected SDK output
        raise AISummaryError("Unexpected response format from OpenAI API") from exc

    bullets = _normalise_output(text)
    if not bullets:
        raise AISummaryError("OpenAI response was empty")

    return bullets


def _normalise_output(response_text: str) -> list[str]:
    normalized: list[str] = []
    for line in response_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped.startswith("• "):
            stripped = stripped[2:].strip()
        normalized.append(stripped)
    return normalized


__all__ = [
    "AISummaryError",
    "AISummaryRequest",
    "build_ai_summary_request",
    "generate_ai_summary",
]

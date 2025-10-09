"""AI-assisted summary generation for PlainSpend."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Union

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
    mode: str = "overview"


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


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_category_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        pct_change_raw = row.get("PctChange")
        pct_change: float | None
        if pct_change_raw is None or pd.isna(pct_change_raw):
            pct_change = None
        else:
            pct_change = _coerce_float(pct_change_raw)

        records.append(
            {
                "category": str(row.get("Category", "")),
                "current": _coerce_float(row.get("CurrentValue", 0.0)),
                "previous": _coerce_float(row.get("PreviousValue", 0.0)),
                "share": _coerce_float(row.get("Share", 0.0)),
                "pct_change": pct_change,
                "change_amount": _coerce_float(row.get("ChangeAmount", 0.0)),
                "rank": _coerce_int(row.get("Rank", 0)),
            }
        )
    return records


def _select_top(items: Iterable[dict[str, Any]], key: str, count: int, descending: bool) -> list[dict[str, Any]]:
    rows = list(items)
    rows.sort(key=lambda record: record.get(key, 0.0) or 0.0, reverse=descending)

    trimmed: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(key)
        if value in (None, 0, 0.0):
            continue
        if descending and value > 0:
            trimmed.append(row)
        elif not descending and value < 0:
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


ProgressInput = Union[ProgressRow, Mapping[str, Any]]


def _extract_subscriptions(progress_rows: Iterable[ProgressInput]) -> list[dict[str, Any]]:
    subs: list[dict[str, Any]] = []
    for row in progress_rows:
        if isinstance(row, Mapping):
            category = str(row.get("category", "")).lower()
            label = str(row.get("label", ""))
            detail = str(row.get("delta", ""))
        else:
            category = str(row.category).lower()  # type: ignore[attr-defined]
            label = str(row.label)  # type: ignore[attr-defined]
            detail = str(row.delta)  # type: ignore[attr-defined]

        if category != "subscriptions":
            continue
        subs.append({"merchant": label, "detail": detail})

    return subs[:5]


def _summarise_runway(runway: Mapping[str, Any] | None) -> dict[str, Any]:
    if not runway:
        return {}

    def _format_date(value: Any) -> str | None:
        if value is None:
            return None
        try:
            return pd.Timestamp(value).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            return str(value)

    return {
        "status": str(runway.get("status", "")),
        "message": str(runway.get("message", "")),
        "cycle_days": _coerce_int(runway.get("cycle_days", 0)),
        "safe_days": _coerce_int(runway.get("days_safe", 0)),
        "days_until_payday": _coerce_int(runway.get("days_until_payday", 0)),
        "average_daily_spend": _coerce_float(runway.get("average_daily_spend", 0.0)),
        "remaining_budget": _coerce_float(runway.get("remaining_budget", 0.0)),
        "upcoming_commitments": _coerce_float(runway.get("upcoming_commitments", 0.0)),
        "projected_spend_rest": _coerce_float(runway.get("projected_spend_rest", 0.0)),
        "forecast_gap": _coerce_float(runway.get("forecast_gap", 0.0)),
        "next_payday": _format_date(runway.get("next_payday")),
        "last_payday": _format_date(runway.get("last_payday")),
    }


def _summarise_recurring(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in entries:
        next_date = entry.get("next_date")
        try:
            next_date_str = pd.Timestamp(next_date).strftime("%Y-%m-%d") if next_date else None
        except (TypeError, ValueError):
            next_date_str = str(next_date) if next_date is not None else None

        items.append(
            {
                "merchant": str(entry.get("merchant", "")),
                "interval": str(entry.get("interval_label", "")),
                "average_amount": _coerce_float(entry.get("average_amount", 0.0)),
                "days_until_due": _coerce_int(entry.get("days_until_due", 0)),
                "next_date": next_date_str,
                "confidence": _coerce_float(entry.get("confidence", 0.0)),
            }
        )
    return items[:5]


def _summarise_duplicates(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in entries:
        date_value = entry.get("date")
        try:
            date_str = pd.Timestamp(date_value).strftime("%Y-%m-%d") if date_value else None
        except (TypeError, ValueError):
            date_str = str(date_value) if date_value is not None else None

        raw_amounts = entry.get("amounts", [])
        amounts: list[float] = []
        for amount in raw_amounts:
            try:
                amounts.append(float(amount))
            except (TypeError, ValueError):
                continue

        items.append(
            {
                "merchant": str(entry.get("merchant", "")),
                "date": date_str,
                "count": _coerce_int(entry.get("count", len(amounts)) or len(amounts)),
                "amounts": amounts,
                "total_amount": _coerce_float(entry.get("total_amount", sum(amounts))),
            }
        )
    return items[:5]


def build_ai_summary_request(data: DashboardData, mode: str = "overview") -> AISummaryRequest:
    """Create a payload ready to be sent to the OpenAI API."""

    safe_mode = mode if mode in {"overview", "insights"} else "overview"

    summary = data["monthly_summary"]
    categories = _build_category_records(data["category_df"])
    risers = _select_top(categories, "pct_change", 3, descending=True)
    fallers = _select_top(categories, "pct_change", 3, descending=False)
    anomalies = _detect_anomalies(categories)

    progress_rows = data.get("progress_rows") or []
    subs = _extract_subscriptions(progress_rows)

    spend_totals = {
        "total": _coerce_float(summary.get("total", 0.0)),
        "delta": str(summary.get("delta", "")),
        "avg_day": _coerce_float(summary.get("avg_day", 0.0)),
        "highest_day": _coerce_float(summary.get("highest_day", 0.0)),
        "projected_total": _coerce_float(summary.get("projected_total", 0.0)),
        "projection_low": _coerce_float(summary.get("projection_low", 0.0)),
        "projection_high": _coerce_float(summary.get("projection_high", 0.0)),
        "projection_confidence": _coerce_float(summary.get("projection_confidence", 0.0)),
        "days_elapsed": _coerce_int(summary.get("days_elapsed", 0)),
        "days_remaining": _coerce_int(summary.get("days_remaining", 0)),
    }

    if safe_mode == "insights":
        payload = {
            "month": summary["month_label"],
            "spend_totals": spend_totals,
            "runway": _summarise_runway(data.get("cashflow_runway")),
            "recurring_watchlist": _summarise_recurring(data.get("recurring_entries", [])),
            "duplicate_flags": _summarise_duplicates(data.get("duplicate_entries", [])),
            "category_focus": {
                "top_risers": risers,
                "top_fallers": fallers,
                "anomalies": anomalies,
            },
        }
    else:
        payload = {
            "month": summary["month_label"],
            "spend_totals": spend_totals,
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

    return AISummaryRequest(
        payload=payload,
        month_label=summary["month_label"],
        model=model,
        mode=safe_mode,
    )


def generate_ai_summary(data: DashboardData, mode: str = "overview") -> list[str]:
    """Generate a list of bullet point insights using OpenAI."""

    request = build_ai_summary_request(data, mode=mode)
    client = _resolve_openai_client()

    if request.mode == "insights":
        system_prompt = (
            "You are a financial planning co-pilot."
            " Focus on cash runway, recurring commitments, and anomalies that need attention."
            " Keep tone proactive yet calm, and reference pound values with the £ symbol."
        )
        guidance = (
            "Craft 3–4 concise bullet points that cover:"
            " 1) the overall runway health and any immediate actions before payday;"
            " 2) recurring charges that merit review, including timing or unusual movement;"
            " 3) duplicate or same-day transactions that might require follow-up;"
            " 4) standout category shifts worth watching."
            " Tie each point to concrete numbers from the JSON."
        )
    else:
        system_prompt = (
            "You are an assistant that writes concise financial summaries for personal spending."
            " You receive structured JSON summarising the current and previous month's card spending."
            " Write 3-4 bullet points comparing current vs last month, highlighting:"
            " (1) major spend changes, (2) categories with notable rises/falls or anomalies,"
            " and (3) subscriptions to review. Use a professional but friendly tone,"
            " UK currency formatting with the pound symbol, and keep it under 120 words."
        )
        guidance = (
            "Summarise the key takeaways from the JSON."
            " Include concrete figures for notable increases or decreases and subscription call-outs."
        )

    formatted_payload = json.dumps(request.payload, ensure_ascii=False, indent=2)
    user_message = (
        f"Analyse the spending context for {request.month_label}.\n"
        f"Guidance: {guidance}\n\n"
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

"""Insights dashboard page layout."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from app.layout import card
from core import DashboardData


def _format_due_phrase(days: int) -> str:
    if days < 0:
        return f"Overdue by {abs(days)} day{'s' if abs(days) != 1 else ''}"
    if days == 0:
        return "Due today"
    if days == 1:
        return "Due tomorrow"
    return f"Due in {days} days"


def _format_date_label(value: Any) -> str:
    if value is None:
        return "TBC"
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if value is None:
            return "TBC"
    if isinstance(value, pd.Index):
        value = value[0] if not value.empty else None
        if value is None:
            return "TBC"
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(timestamp):
        return "TBC"
    return timestamp.strftime("%d %b")


def _render_insights_card(insights: list[str]) -> None:
    items = "".join(f"<li>{item}</li>" for item in insights)
    st.markdown(
        f"<ul class='ps-insights'>{items}</ul>",
        unsafe_allow_html=True,
    )


def _render_runway_card(runway: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    if not runway:
        st.info("Runway insights are still loading.")
        return

    safe_days = int(runway.get("days_safe", 0))
    days_until_payday = max(int(runway.get("days_until_payday", 0)), 0)
    ratio = 1.0 if days_until_payday == 0 else safe_days / max(days_until_payday, 1)
    ratio = float(min(max(ratio, 0.0), 1.0))

    progress_text = f"{safe_days} safe days · {days_until_payday} until payday"
    st.progress(ratio, text=progress_text)

    meta_cols = st.columns(2)
    meta_cols[0].metric("Next payday", _format_date_label(runway.get("next_payday")))
    meta_cols[1].metric("Burn rate", f"£{runway.get('average_daily_spend', 0):,.0f}/day")

    budget_cols = st.columns(2)
    budget_cols[0].metric("Commitments", f"£{runway.get('upcoming_commitments', 0):,.0f}")
    budget_cols[1].metric("Buffer", f"£{runway.get('remaining_budget', 0):,.0f}")

    st.caption(runway.get("message", ""))

    detail_items = [
        f"Total spend to date · £{summary['total']:,.0f}",
        f"Projection gap · £{runway.get('forecast_gap', 0):,.0f}",
    ]
    st.markdown(
        f"<ul class='ps-insights'>{''.join(f'<li>{item}</li>' for item in detail_items)}</ul>",
        unsafe_allow_html=True,
    )


def _render_recurring_list(entries: Sequence[Mapping[str, Any]]) -> None:
    if not entries:
        st.success("No recurring subscriptions detected yet.")
        return

    items: list[str] = []
    for entry in entries[:6]:
        due_phrase = _format_due_phrase(int(entry.get("days_until_due", 0)))
        next_label = _format_date_label(entry.get("next_date"))
        avg_amount = float(entry.get("average_amount", 0.0))
        interval_label = str(entry.get("interval_label", "Recurring"))
        merchant = entry.get("merchant", "Unknown")
        items.append(
            f"<li><strong>{merchant}</strong> · £{avg_amount:,.2f} {interval_label.lower()}"
            f" · {due_phrase} (next {next_label})</li>"
        )

    st.markdown(
        f"<ul class='ps-insights'>{''.join(items)}</ul>",
        unsafe_allow_html=True,
    )


def _render_duplicate_list(entries: Sequence[Mapping[str, Any]]) -> None:
    if not entries:
        st.success("No duplicate charges spotted this month.")
        return

    items: list[str] = []
    for entry in entries[:5]:
        merchant = entry.get("merchant", "Unknown")
        date_label = _format_date_label(entry.get("date"))
        amounts = entry.get("amounts", [])
        amount_summary = ", ".join(f"£{float(amount):,.2f}" for amount in amounts)
        count = int(entry.get("count", len(amounts)))
        items.append(
            f"<li><strong>{merchant}</strong> · {count} transactions on {date_label}<br />"
            f"<span style='color:#4B5563'>Amounts: {amount_summary}</span></li>"
        )

    st.markdown(
        f"<ul class='ps-insights'>{''.join(items)}</ul>",
        unsafe_allow_html=True,
    )


def _render_action_checklist(
    cashflow: Mapping[str, Any],
    recurring_entries: Sequence[Mapping[str, Any]],
    duplicate_entries: Sequence[Mapping[str, Any]],
) -> None:
    action_items: list[str] = []

    if cashflow:
        status = "Stay on pace" if cashflow.get("status") == "on_track" else "Tighten spend"
        action_items.append(f"{status}: {cashflow.get('message', '')}")

    if recurring_entries:
        top = recurring_entries[0]
        due_phrase = _format_due_phrase(int(top.get("days_until_due", 0)))
        amount = float(top.get("average_amount", 0.0))
        cadence = str(top.get("interval_label", "Recurring")).lower()
        action_items.append(
            f"Review {top.get('merchant', 'subscription')} (£{amount:,.2f} {cadence}) — {due_phrase}."
        )

    if duplicate_entries:
        dup = duplicate_entries[0]
        count = int(dup.get("count", len(dup.get("amounts", []))))
        action_items.append(
            f"Double-check {dup.get('merchant', 'recent spend')} duplicates on {_format_date_label(dup.get('date'))} "
            f"({count} charges)."
        )

    if not action_items:
        st.success("All clear—no immediate follow-ups detected.")
    else:
        st.markdown(
            f"<ul class='ps-insights'>{''.join(f'<li>{item}</li>' for item in action_items)}</ul>",
            unsafe_allow_html=True,
        )


def render_page(data: DashboardData, ai_insights: list[str]) -> None:
    """Render the insights dashboard page."""

    st.title("Insights")
    st.caption("Deep dives, AI narratives, and areas to watch this month.")

    summary = data["monthly_summary"]
    cashflow = data.get("cashflow_runway", {})
    recurring_entries = data.get("recurring_entries", [])
    duplicate_entries = data.get("duplicate_entries", [])

    highlights_col, runway_col = st.columns((2, 1.2), gap="medium")
    with highlights_col:
        with card("AI highlights", suffix="Generated"):
            if ai_insights:
                _render_insights_card(ai_insights)
            else:
                st.info("AI highlights will appear here once available.")

    with runway_col:
        with card("Cashflow runway", suffix="Payday aware"):
            _render_runway_card(cashflow, summary)

    recurring_col, duplicate_col = st.columns((2, 1), gap="medium")
    with recurring_col:
        with card("Recurring watchlist", suffix="Subscriptions & bills"):
            _render_recurring_list(recurring_entries)

    with duplicate_col:
        with card("Potential duplicates", suffix="Same-day spend"):
            _render_duplicate_list(duplicate_entries)

    with card("Category movers", suffix="Top shifts"):
        category_df = data["category_df"]
        if category_df.empty:
            st.info("Once category spend is available, you'll see the biggest movers here.")
        else:
            movers = (
                category_df[["Category", "CurrentValue", "ChangeAmount", "PctChange"]]
                .sort_values("ChangeAmount", ascending=False)
                .head(5)
                .copy()
            )
            items = []
            for _, row in movers.iterrows():
                pct_change = row["PctChange"]
                pct_display = (
                    f" ({pct_change:+.1%})"
                    if pct_change is not None and not pd.isna(pct_change)
                    else ""
                )
                items.append(
                    f"<li><strong>{row['Category']}</strong> · £{row['CurrentValue']:,.0f}"
                    f" ({row['ChangeAmount']:+,.0f}{pct_display})</li>"
                )
            st.markdown(
                f"<ul class='ps-insights'>{''.join(items)}</ul>",
                unsafe_allow_html=True,
            )

    with card("Action checklist", suffix="Next steps"):
        _render_action_checklist(cashflow, recurring_entries, duplicate_entries)


__all__ = ["render_page"]

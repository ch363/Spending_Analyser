"""Unit tests for key analytics helpers and the summary service."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.summary import prepare_dashboard_data
from core.ai.summary import build_ai_summary_request
from analytics.categorisation import prepare_expenses
from analytics.forecasting import build_daily_spend, compute_cashflow_runway


@pytest.fixture(autouse=True)
def clear_streamlit_secrets(monkeypatch):
    """Provide an empty secrets mapping so tests don't rely on Streamlit runtime."""

    monkeypatch.setattr(st, "secrets", {}, raising=False)


@pytest.fixture()
def sample_transactions_frame() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "date": "2023-12-10",
                "description": "Groceries R US",
                "amount": -40.0,
                "category": "groceries",
                "txn_id": "dec-1",
                "note": "seed",
                "is_refund": False,
            },
            {
                "date": "2024-01-01",
                "description": "Groceries R US",
                "amount": -50.0,
                "category": "groceries",
                "txn_id": "jan-1",
                "note": "seed",
                "is_refund": False,
            },
            {
                "date": "2024-01-02",
                "description": "City Transport",
                "amount": -30.0,
                "category": "transport",
                "txn_id": "jan-2",
                "note": "seed",
                "is_refund": False,
            },
            {
                "date": "2024-01-03",
                "description": "Video Stream",
                "amount": -20.0,
                "category": "subscriptions",
                "txn_id": "jan-3",
                "note": "seed",
                "is_refund": False,
            },
            {
                "date": "2024-01-04",
                "description": "Video Stream",
                "amount": -20.0,
                "category": "subscriptions",
                "txn_id": "jan-4",
                "note": "seed",
                "is_refund": False,
            },
            {
                "date": "2024-01-05",
                "description": "Employer Ltd",
                "amount": 2000.0,
                "category": "income",
                "txn_id": "jan-5",
                "note": "seed",
                "is_refund": False,
            },
        ]
    )
    raw["date"] = pd.to_datetime(raw["date"])
    return raw


@pytest.fixture()
def sample_transactions_csv(sample_transactions_frame, tmp_path) -> str:
    csv_path = tmp_path / "transactions.csv"
    sample_transactions_frame.to_csv(csv_path, index=False)
    return str(csv_path)


def test_build_daily_spend_enforces_continuous_range(sample_transactions_frame):
    expenses = prepare_expenses(sample_transactions_frame)
    month_start = pd.Timestamp("2024-01-01")
    current_day = pd.Timestamp("2024-01-05")

    series = build_daily_spend(expenses, month_start, current_day)

    assert series.index[0] == month_start
    assert series.index[-1] == current_day
    assert len(series) == 5
    assert series.loc[pd.Timestamp("2024-01-01")] == pytest.approx(50.0)
    assert series.loc[pd.Timestamp("2024-01-03")] == pytest.approx(20.0)
    assert series.loc[pd.Timestamp("2024-01-05")] == pytest.approx(0.0)


def test_compute_cashflow_runway_budgeting():
    expenses = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-06", "2024-01-08"]),
            "spend": [75.0, 45.0, 30.0],
        }
    )
    incomes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "amount": [3000.0],
        }
    )

    runway = compute_cashflow_runway(
        expenses=expenses,
        incomes=incomes,
        today=pd.Timestamp("2024-01-10"),
        total_spend_to_date=150.0,
        projected_total=420.0,
        days_remaining=21,
        recurring_entries=[
            {
                "group_key": "video-stream",
                "merchant": "Video Stream",
                "interval_days": 30,
                "interval_label": "Monthly",
                "average_amount": 30.0,
                "last_amount": 30.0,
                "last_date": pd.Timestamp("2023-12-12"),
                "next_date": pd.Timestamp("2024-01-12"),
                "days_until_due": 2,
                "occurrences": 6,
                "confidence": 0.9,
            }
        ],
    )

    assert runway["status"] == "on_track"
    assert runway["days_until_payday"] == 21
    assert runway["days_safe"] == 21
    assert runway["remaining_budget"] == pytest.approx(2820.0)
    assert "You're on track" in runway["message"]


def test_prepare_dashboard_data_basic(sample_transactions_csv):
    dashboard_data = prepare_dashboard_data(
        sample_transactions_csv,
        target_date=date(2024, 1, 15),
        confidence=0.0,
    )

    monthly_summary = dashboard_data["monthly_summary"]
    assert monthly_summary["total"] == pytest.approx(120.0)
    assert monthly_summary["delta"].startswith("+")
    assert monthly_summary["subscriptions"] == pytest.approx(40.0)
    assert monthly_summary["days_elapsed"] > 0

    category_df = dashboard_data["category_df"]
    assert not category_df.empty
    assert "Category" in category_df.columns


def test_build_ai_summary_request_modes(sample_transactions_csv):
    dashboard_data = prepare_dashboard_data(
        sample_transactions_csv,
        target_date=date(2024, 1, 15),
        confidence=0.0,
    )

    overview = build_ai_summary_request(dashboard_data, mode="overview")
    assert overview.mode == "overview"
    assert "category_changes" in overview.payload
    assert "subscriptions" in overview.payload

    insights = build_ai_summary_request(dashboard_data, mode="insights")
    assert insights.mode == "insights"
    assert "runway" in insights.payload
    assert "category_focus" in insights.payload
    assert "top_risers" in insights.payload["category_focus"]

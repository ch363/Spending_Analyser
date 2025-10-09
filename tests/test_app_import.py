import importlib
from typing import cast
from types import SimpleNamespace

import pandas as pd
from openai import OpenAI

from core import DashboardData, generate_ai_summary


def test_app_package_exports_main():
    module = importlib.import_module("app")

    assert hasattr(module, "main"), "app package should expose main entrypoint"


def test_generate_ai_summary_uses_injected_client():
    payload: DashboardData = {
        "monthly_summary": {
            "month_label": "January 2025",
            "total": 1000.0,
            "delta": "+10%",
            "avg_day": 100.0,
            "highest_day": 200.0,
            "subscriptions": 300.0,
            "projected_total": 1200.0,
            "projection_low": 1100.0,
            "projection_high": 1300.0,
            "projection_confidence": 0.68,
            "days_elapsed": 10,
            "days_remaining": 20,
        },
    "daily_spend_df": pd.DataFrame(),
    "cumulative_spend_df": pd.DataFrame(),
    "category_df": pd.DataFrame(),
        "progress_rows": [],
        "vendor_rows": [],
        "insights": [],
        "recurring_entries": [],
        "duplicate_entries": [],
        "cashflow_runway": {
            "cycle_days": 30,
            "last_payday": None,
            "next_payday": None,
            "days_until_payday": 10,
            "days_safe": 5,
            "average_daily_spend": 40.0,
            "remaining_budget": 200.0,
            "upcoming_commitments": 100.0,
            "projected_spend_rest": 400.0,
            "forecast_gap": 200.0,
            "message": "All good",
            "status": "on_track",
        },
    }

    class DummyClient:
        def __init__(self):
            self.called = False

        class Chat:
            def __init__(self, outer):
                self.outer = outer

            class Completions:
                def __init__(self, outer):
                    self.outer = outer

                def create(self, **_: object):
                    self.outer.outer.called = True
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="- Bullet"))]
                    )

            @property
            def completions(self):
                return DummyClient.Chat.Completions(self)

        @property
        def chat(self):
            return DummyClient.Chat(self)

    client = DummyClient()

    result = generate_ai_summary(
        payload,
        client_factory=lambda: cast(OpenAI, client),
    )

    assert client.called, "Injected client should be used for AI summary generation"
    assert result == ["Bullet"]

"""Cash-flow projection helpers for insights experiences."""

from __future__ import annotations

from typing import Iterable, TypedDict

import numpy as np
import pandas as pd

from .recurring import RecurringEntry

class CashflowRunway(TypedDict):
	cycle_days: int
	last_payday: pd.Timestamp | None
	next_payday: pd.Timestamp | None
	days_until_payday: int
	days_safe: int
	average_daily_spend: float
	remaining_budget: float
	upcoming_commitments: float
	projected_spend_rest: float
	forecast_gap: float
	message: str
	status: str


def compute_cashflow_runway(
	expenses: pd.DataFrame,
	incomes: pd.DataFrame,
	today: pd.Timestamp,
	total_spend_to_date: float,
	projected_total: float,
	days_remaining: int,
	recurring_entries: Iterable[RecurringEntry],
) -> CashflowRunway:
	"""Estimate spend runway until the next payday using historic behaviour."""

	today = pd.Timestamp(today).normalize()
	pay_dates = incomes.loc[incomes["amount"] > 0, "date"].dropna().dt.normalize().sort_values()

	cycle_days = _infer_cycle_days(pay_dates)
	last_payday = _resolve_last_payday(pay_dates, today)
	next_payday = _resolve_next_payday(last_payday, cycle_days, today)

	days_until_payday = int((next_payday - today).days) if next_payday is not None else max(days_remaining, 0)
	days_until_payday = max(days_until_payday, 0)

	cycle_start = last_payday if last_payday is not None else today - pd.Timedelta(days=cycle_days)
	cycle_income = _resolve_cycle_income(incomes, last_payday) if last_payday is not None else incomes["amount"].clip(lower=0).median()
	if pd.isna(cycle_income) or cycle_income <= 0:
		cycle_income = float(incomes["amount"].clip(lower=0).mean() or 0.0)

	cycle_mask = (expenses["date"].dt.normalize() > cycle_start) & (expenses["date"].dt.normalize() <= today)
	cycle_spend = float(expenses.loc[cycle_mask, "spend"].sum())

	days_elapsed = max((today - cycle_start).days, 1)
	average_daily_spend = cycle_spend / days_elapsed

	commitments = _sum_upcoming_commitments(recurring_entries, today, next_payday)

	remaining_budget = float((cycle_income or 0.0) - cycle_spend - commitments)

	if average_daily_spend <= 0:
		days_safe = days_until_payday
	else:
		days_safe = int(max(min(remaining_budget / average_daily_spend, days_until_payday), 0))

	projected_spend_rest = average_daily_spend * days_until_payday
	forecast_gap = float(projected_total - total_spend_to_date)
	status = "on_track" if remaining_budget >= 0 else "at_risk"

	message = _build_runway_message(days_safe, days_until_payday, status)

	return {
		"cycle_days": cycle_days,
		"last_payday": last_payday,
		"next_payday": next_payday,
		"days_until_payday": days_until_payday,
		"days_safe": days_safe,
		"average_daily_spend": float(round(average_daily_spend, 2)),
		"remaining_budget": float(round(remaining_budget, 2)),
		"upcoming_commitments": float(round(commitments, 2)),
		"projected_spend_rest": float(round(projected_spend_rest, 2)),
		"forecast_gap": float(round(forecast_gap, 2)),
		"message": message,
		"status": status,
	}


def _infer_cycle_days(pay_dates: pd.Series) -> int:
	if pay_dates.empty:
		return 30

	diffs = pay_dates.diff().dt.days.dropna()
	if diffs.empty:
		return 30

	median_diff = float(np.median(diffs))
	if np.isnan(median_diff) or median_diff <= 0:
		return 30

	if median_diff <= 9:
		return 7
	if median_diff <= 21:
		return 14
	return 30


def _resolve_last_payday(pay_dates: pd.Series, today: pd.Timestamp) -> pd.Timestamp | None:
	if pay_dates.empty:
		return None
	past = pay_dates[pay_dates <= today]
	if not past.empty:
		return pd.Timestamp(past.iloc[-1])
	return pd.Timestamp(pay_dates.iloc[0])


def _resolve_next_payday(
	last_payday: pd.Timestamp | None,
	cycle_days: int,
	today: pd.Timestamp,
) -> pd.Timestamp | None:
	if last_payday is None:
		return today + pd.Timedelta(days=cycle_days)

	next_payday = last_payday + pd.Timedelta(days=cycle_days)
	while next_payday <= today:
		next_payday += pd.Timedelta(days=cycle_days)
	return next_payday


def _resolve_cycle_income(incomes: pd.DataFrame, last_payday: pd.Timestamp | None) -> float:
	if last_payday is None:
		return float(incomes["amount"].clip(lower=0).median())

	mask = incomes["date"].dt.normalize() == last_payday
	income_value = float(incomes.loc[mask, "amount"].sum())
	if income_value > 0:
		return income_value
	return float(incomes["amount"].clip(lower=0).median())


def _sum_upcoming_commitments(
	recurring_entries: Iterable[RecurringEntry],
	today: pd.Timestamp,
	next_payday: pd.Timestamp | None,
) -> float:
	total = 0.0
	for entry in recurring_entries:
		next_date = entry.get("next_date")
		if next_date is None:
			continue
		next_date = pd.Timestamp(next_date).normalize()
		if next_date < today:
			continue
		if next_payday is not None and next_date > next_payday:
			continue
		total += float(entry.get("average_amount", 0.0))
	return total


def _build_runway_message(days_safe: int, days_until_payday: int, status: str) -> str:
	if days_until_payday <= 0:
		return "Payday has arrived—reassess your commitments for the new cycle."

	tone_prefix = "You're on track" if status == "on_track" else "Heads up"
	return (
		f"{tone_prefix}: about {days_safe} safe days remain out of the next {days_until_payday} before payday"
	)

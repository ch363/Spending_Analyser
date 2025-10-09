"""Time-series forecasting and cashflow analytics helpers."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Tuple, TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd

from core.models import DailyForecastPoint, ProjectionResult

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from analytics.recurring import RecurringEntry

__all__ = [
    "build_daily_spend",
    "compute_projection",
    "build_daily_and_cumulative_frames",
    "resolve_target_period",
    "resolve_current_day",
    "CashflowRunway",
    "compute_cashflow_runway",
]


class CashflowRunway(TypedDict):
    """Summary of cash runway conditions leading up to the next payday."""

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


def resolve_target_period(expenses: pd.DataFrame, target_date: date | None) -> pd.Period:
    """Return the reporting month derived from expenses or an explicit target."""

    if target_date is not None:
        return pd.Period(target_date, freq="M")
    latest_date = expenses["date"].max()
    return latest_date.to_period("M")


def resolve_current_day(month_df: pd.DataFrame, month_end: pd.Timestamp) -> pd.Timestamp:
    """Return the most recent observed transaction day within the month."""

    if month_df.empty:
        return month_end
    observed_max = month_df["date"].max().normalize()
    return min(observed_max, month_end)


def build_daily_spend(
    expenses: pd.DataFrame,
    month_start: pd.Timestamp,
    current_day: pd.Timestamp,
) -> pd.Series:
    """Construct a daily spend series between ``month_start`` and ``current_day``."""

    if current_day < month_start:
        return pd.Series(dtype=float)

    index = pd.date_range(month_start, current_day, freq="D")
    grouped = (
        expenses.groupby(pd.Grouper(key="date", freq="D"))["spend"].sum().reindex(index, fill_value=0.0)
    )
    grouped = grouped.astype(float)
    grouped.index.name = "Day"
    return grouped


def compute_projection(
    daily_spend: pd.Series,
    current_day: pd.Timestamp,
    month_end: pd.Timestamp,
    confidence: float,
) -> ProjectionResult:
    """Forecast cumulative spend for the remainder of the month."""

    total_to_date = float(daily_spend.sum())
    days_elapsed = int(len(daily_spend))
    daily_points: list[DailyForecastPoint] = []

    if current_day >= month_end:
        return ProjectionResult(total_to_date, total_to_date, total_to_date, 0, days_elapsed, daily_points)

    future_days = pd.date_range(current_day + pd.Timedelta(days=1), month_end, freq="D")
    if future_days.empty:
        return ProjectionResult(total_to_date, total_to_date, total_to_date, 0, days_elapsed, daily_points)

    weekday_frame = daily_spend.to_frame(name="spend")
    weekday_index = pd.Series(weekday_frame.index, index=weekday_frame.index)
    weekday_frame["weekday"] = weekday_index.dt.weekday

    weekday_means = weekday_frame.groupby("weekday")["spend"].mean()
    weekday_stds = weekday_frame.groupby("weekday")["spend"].std(ddof=0)
    overall_mean = weekday_frame["spend"].mean()
    overall_std = weekday_frame["spend"].std(ddof=0)

    if np.isnan(overall_mean):
        overall_mean = 0.0
    if np.isnan(overall_std):
        overall_std = 0.0

    additional_mean = 0.0
    variance = 0.0
    for day in future_days:
        weekday = day.weekday()
        mu = weekday_means.get(weekday, np.nan)
        sigma = weekday_stds.get(weekday, np.nan)

        if np.isnan(mu):
            mu = overall_mean
        if np.isnan(sigma):
            sigma = overall_std

        additional_mean += float(mu)
        variance += float(sigma) ** 2
        daily_points.append(DailyForecastPoint(date=day, mean=float(mu), std=float(sigma)))

    projected_total = total_to_date + additional_mean
    if variance <= 0:
        return ProjectionResult(projected_total, projected_total, projected_total, len(future_days), days_elapsed, daily_points)

    from statistics import NormalDist

    z = NormalDist().inv_cdf((1 + confidence) / 2)
    std_future = float(np.sqrt(variance))
    low = max(projected_total - z * std_future, total_to_date)
    high = projected_total + z * std_future
    return ProjectionResult(projected_total, low, high, len(future_days), days_elapsed, daily_points)


def build_daily_and_cumulative_frames(
    daily_spend: pd.Series,
    projection: ProjectionResult,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return data frames for daily spend and cumulative projections."""

    daily_records: list[dict[str, object]] = []
    cumulative_records: list[dict[str, object]] = []

    for day, value in daily_spend.items():
        amount = float(value)
        daily_records.append({"Day": day, "Spend": amount, "Series": "Actual"})

    cumulative_actual = daily_spend.cumsum()
    for day, value in cumulative_actual.items():
        cumulative_records.append({"Day": day, "Total": float(value), "Series": "Actual"})

    if projection.daily_forecast:
        if not daily_spend.empty:
            anchor_day = daily_spend.index[-1]
            anchor_daily = float(daily_spend.iloc[-1])
            daily_records.append({"Day": anchor_day, "Spend": anchor_daily, "Series": "Projected"})
            anchor_total = float(cumulative_actual.iloc[-1])
            cumulative_records.append({"Day": anchor_day, "Total": anchor_total, "Series": "Projected"})
        else:
            anchor_total = 0.0

        running_total = anchor_total
        for point in projection.daily_forecast:
            daily_records.append({"Day": point.date, "Spend": float(point.mean), "Series": "Projected"})
            running_total += float(point.mean)
            cumulative_records.append({"Day": point.date, "Total": running_total, "Series": "Projected"})

    daily_df = pd.DataFrame(daily_records)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("Day")

    cumulative_df = pd.DataFrame(cumulative_records)
    if not cumulative_df.empty:
        cumulative_df = cumulative_df.sort_values("Day")

    return daily_df, cumulative_df


def compute_cashflow_runway(
    expenses: pd.DataFrame,
    incomes: pd.DataFrame,
    today: pd.Timestamp,
    total_spend_to_date: float,
    projected_total: float,
    days_remaining: int,
    recurring_entries: Iterable["RecurringEntry"],
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
    average_daily_spend = cycle_spend / days_elapsed if days_elapsed else 0.0

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
    recurring_entries: Iterable["RecurringEntry"],
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

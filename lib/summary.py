"""Utilities for preparing dashboard data from the spending CSV."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist
from typing import Optional, Tuple, TypedDict

import numpy as np
import pandas as pd


class ProgressRow(TypedDict):
    category: str
    label: str
    delta: str
    width: int


class VendorRow(TypedDict):
    category: str
    label: str
    amount: float


class MonthlySummary(TypedDict):
    total: float
    delta: str
    avg_day: float
    highest_day: float
    subscriptions: float
    projected_total: float
    projection_low: float
    projection_high: float
    projection_confidence: float
    days_elapsed: int
    days_remaining: int
    month_label: str


class DashboardData(TypedDict):
    daily_spend_df: pd.DataFrame
    cumulative_spend_df: pd.DataFrame
    category_df: pd.DataFrame
    progress_rows: list[ProgressRow]
    vendor_rows: list[VendorRow]
    insights: list[str]
    monthly_summary: MonthlySummary


@dataclass(frozen=True)
class ProjectionResult:
    projected_total: float
    low: float
    high: float
    days_remaining: int
    days_elapsed: int
    daily_forecast: list["DailyForecastPoint"]


@dataclass(frozen=True)
class DailyForecastPoint:
    date: pd.Timestamp
    mean: float
    std: float


def prepare_dashboard_data(
    csv_path: str | Path,
    target_date: Optional[date] = None,
    confidence: float = 0.68,
) -> DashboardData:
    """Create dashboard-friendly aggregates from a transactions CSV."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = _load_transactions(csv_path)
    if df.empty:
        raise ValueError("The provided CSV contains no transactions.")

    expenses = _prepare_expenses(df)
    if expenses.empty:
        raise ValueError("No spend transactions available after filtering income and duplicates.")

    target_period = _resolve_target_period(expenses, target_date)
    previous_period = target_period - 1

    current_month_all = df[df["date"].dt.to_period("M") == target_period]
    current_month_expenses = expenses[expenses["date"].dt.to_period("M") == target_period]
    previous_month_expenses = expenses[expenses["date"].dt.to_period("M") == previous_period]

    month_start = target_period.to_timestamp(how="start")
    month_end = target_period.to_timestamp(how="end")
    current_day = _resolve_current_day(current_month_all, month_end)

    daily_spend = _build_daily_spend(current_month_expenses, month_start, current_day)
    projection = _compute_projection(daily_spend, current_day, month_end, confidence)
    daily_spend_df, cumulative_spend_df = _build_daily_and_cumulative_frames(
        daily_spend, projection
    )

    total_spend = float(daily_spend.sum())
    prev_total = float(previous_month_expenses["spend"].sum())
    avg_day = float(total_spend / projection.days_elapsed) if projection.days_elapsed else 0.0
    highest_day = float(daily_spend.max()) if not daily_spend.empty else 0.0
    subscriptions_total = _compute_category_total(current_month_expenses, "subscriptions")

    delta_label = _format_delta(total_spend, prev_total)

    category_df = _build_category_breakdown(current_month_expenses, previous_month_expenses)
    progress_rows = _build_progress_rows(current_month_expenses, total_spend)
    vendor_rows = _build_vendor_rows(current_month_expenses)
    insights = _build_insights(
        total_spend=total_spend,
        delta_label=delta_label,
        projection=projection,
        confidence=confidence,
        top_category=category_df.iloc[0] if not category_df.empty else None,
        top_vendor=vendor_rows[0] if vendor_rows else None,
    )

    month_label = target_period.strftime("%B %Y")

    summary: MonthlySummary = {
        "total": total_spend,
        "delta": delta_label,
        "avg_day": avg_day,
        "highest_day": highest_day,
        "subscriptions": subscriptions_total,
        "projected_total": projection.projected_total,
        "projection_low": projection.low,
        "projection_high": projection.high,
        "projection_confidence": confidence,
        "days_elapsed": projection.days_elapsed,
        "days_remaining": projection.days_remaining,
        "month_label": month_label,
    }

    return {
        "daily_spend_df": daily_spend_df,
        "cumulative_spend_df": cumulative_spend_df,
        "category_df": category_df,
        "progress_rows": progress_rows,
        "vendor_rows": vendor_rows,
        "insights": insights,
        "monthly_summary": summary,
    }


def _load_transactions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[~df["txn_id"].str.contains("_dup", case=False, na=False)]
    df = df[~df["note"].str.contains("duplicate", case=False, na=False)]
    df["category"] = df["category"].fillna("uncategorized")
    df["is_refund"] = df["is_refund"].fillna(False)
    return df


def _prepare_expenses(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["category"].str.lower() != "income"].copy()
    expenses["category_label"] = expenses["category"].str.replace("_", " ").str.title()
    expenses["spend"] = np.where(expenses["amount"] < 0, -expenses["amount"], 0.0)
    refund_mask = expenses["is_refund"] & (expenses["amount"] > 0)
    expenses.loc[refund_mask, "spend"] = -expenses.loc[refund_mask, "amount"]
    return expenses


def _resolve_target_period(expenses: pd.DataFrame, target_date: Optional[date]) -> pd.Period:
    if target_date is not None:
        return pd.Period(target_date, freq="M")
    latest_date = expenses["date"].max()
    return latest_date.to_period("M")


def _resolve_current_day(month_df: pd.DataFrame, month_end: pd.Timestamp) -> pd.Timestamp:
    if month_df.empty:
        return month_end
    observed_max = month_df["date"].max().normalize()
    return min(observed_max, month_end)


def _build_daily_spend(
	expenses: pd.DataFrame,
	month_start: pd.Timestamp,
	current_day: pd.Timestamp,
) -> pd.Series:
	if current_day < month_start:
		return pd.Series(dtype=float)

	index = pd.date_range(month_start, current_day, freq="D")
	grouped = (
		expenses.groupby(pd.Grouper(key="date", freq="D"))["spend"].sum().reindex(index, fill_value=0.0)
	)
	grouped = grouped.astype(float)
	grouped.index.name = "Day"
	return grouped


def _compute_projection(
    daily_spend: pd.Series,
    current_day: pd.Timestamp,
    month_end: pd.Timestamp,
    confidence: float,
) -> ProjectionResult:
    total_to_date = float(daily_spend.sum())
    days_elapsed = int(len(daily_spend))
    daily_points: list[DailyForecastPoint] = []

    if current_day >= month_end:
        return ProjectionResult(
            total_to_date,
            total_to_date,
            total_to_date,
            0,
            days_elapsed,
            daily_points,
        )

    future_days = pd.date_range(current_day + pd.Timedelta(days=1), month_end, freq="D")
    if future_days.empty:
        return ProjectionResult(
            total_to_date,
            total_to_date,
            total_to_date,
            0,
            days_elapsed,
            daily_points,
        )

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
        daily_points.append(
            DailyForecastPoint(date=day, mean=float(mu), std=float(sigma))
        )

    projected_total = total_to_date + additional_mean
    if variance <= 0:
        return ProjectionResult(
            projected_total,
            projected_total,
            projected_total,
            len(future_days),
            days_elapsed,
            daily_points,
        )

    z = NormalDist().inv_cdf((1 + confidence) / 2)
    std_future = float(np.sqrt(variance))
    low = max(projected_total - z * std_future, total_to_date)
    high = projected_total + z * std_future
    return ProjectionResult(projected_total, low, high, len(future_days), days_elapsed, daily_points)


def _compute_category_total(expenses: pd.DataFrame, category_name: str) -> float:
    mask = expenses["category"].str.lower() == category_name.lower()
    if not mask.any():
        return 0.0
    total = expenses.loc[mask, "spend"].sum()
    return float(max(total, 0.0))


def _build_category_breakdown(
    current_expenses: pd.DataFrame,
    previous_expenses: pd.DataFrame,
) -> pd.DataFrame:
    if current_expenses.empty:
        return pd.DataFrame(
            columns=[
                "Category",
                "CurrentValue",
                "PreviousValue",
                "Share",
                "PctChange",
                "ChangeAmount",
                "Rank",
            ]
        )

    current_totals = (
        current_expenses.groupby("category_label")["spend"].sum().sort_values(ascending=False)
    )
    current_totals = current_totals[current_totals > 0]

    previous_totals = previous_expenses.groupby("category_label")["spend"].sum()

    top_categories = current_totals.head(8)
    total_value = float(current_totals.sum()) if not current_totals.empty else 0.0

    breakdown = top_categories.reset_index().rename(
        columns={"category_label": "Category", "spend": "CurrentValue"}
    )
    breakdown["PreviousValue"] = breakdown["Category"].map(previous_totals).fillna(0.0)

    current_values = breakdown["CurrentValue"].astype(float)
    prev_values = breakdown["PreviousValue"].astype(float)

    if total_value > 0:
        breakdown["Share"] = current_values / total_value
    else:
        breakdown["Share"] = 0.0

    pct_change = np.where(prev_values > 0, (current_values - prev_values) / prev_values, np.nan)
    breakdown["PctChange"] = pct_change
    breakdown["ChangeAmount"] = current_values - prev_values
    breakdown["Rank"] = np.arange(1, len(breakdown) + 1)

    return breakdown


def _build_progress_rows(expenses: pd.DataFrame, total_spend: float) -> list[ProgressRow]:
    rows: list[ProgressRow] = []
    if total_spend <= 0:
        return rows

    for category, category_df in expenses.groupby("category_label"):
        category_total = category_df["spend"].sum()
        if category_total <= 0:
            continue

        merchant_totals = (
            category_df.groupby("description")["spend"].sum().sort_values(ascending=False)
        )

        for merchant, amount in merchant_totals.head(3).items():
            if amount <= 0:
                continue
            share = (amount / category_total) * 100
            width = int(np.clip(round((amount / category_total) * 100), 0, 100))
            rows.append(
                {
                    "category": str(category),
                    "label": str(merchant),
                    "delta": f"{share:.0f}% of category",
                    "width": width,
                }
            )

    return rows


def _build_vendor_rows(expenses: pd.DataFrame) -> list[VendorRow]:
    merchant_totals = (
        expenses.groupby(["category_label", "description"])["spend"].sum().reset_index()
    )
    merchant_totals = merchant_totals[merchant_totals["spend"] > 0]
    if merchant_totals.empty:
        return []

    merchant_totals = merchant_totals.sort_values("spend", ascending=False)
    top_per_category = (
        merchant_totals.groupby("category_label", group_keys=False).head(5)
    )

    vendor_rows: list[VendorRow] = []
    for record in top_per_category.to_dict(orient="records"):
        vendor_rows.append(
            {
                "category": str(record["category_label"]),
                "label": str(record["description"]),
                "amount": float(record["spend"]),
            }
        )

    return vendor_rows


def _build_daily_and_cumulative_frames(
    daily_spend: pd.Series, projection: ProjectionResult
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_records: list[dict[str, object]] = []
    cumulative_records: list[dict[str, object]] = []

    for day, value in daily_spend.items():
        amount = float(value)
        daily_records.append({"Day": day, "Spend": amount, "Series": "Actual"})

    cumulative_actual = daily_spend.cumsum()
    for day, value in cumulative_actual.items():
        cumulative_records.append(
            {"Day": day, "Total": float(value), "Series": "Actual"}
        )

    if projection.daily_forecast:
        if not daily_spend.empty:
            anchor_day = daily_spend.index[-1]
            anchor_daily = float(daily_spend.iloc[-1])
            daily_records.append(
                {"Day": anchor_day, "Spend": anchor_daily, "Series": "Projected"}
            )
            anchor_total = float(cumulative_actual.iloc[-1])
            cumulative_records.append(
                {"Day": anchor_day, "Total": anchor_total, "Series": "Projected"}
            )
        else:
            anchor_total = 0.0

        running_total = anchor_total
        for point in projection.daily_forecast:
            daily_records.append(
                {
                    "Day": point.date,
                    "Spend": float(point.mean),
                    "Series": "Projected",
                }
            )
            running_total += float(point.mean)
            cumulative_records.append(
                {
                    "Day": point.date,
                    "Total": running_total,
                    "Series": "Projected",
                }
            )

    daily_df = pd.DataFrame(daily_records)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("Day")

    cumulative_df = pd.DataFrame(cumulative_records)
    if not cumulative_df.empty:
        cumulative_df = cumulative_df.sort_values("Day")

    return daily_df, cumulative_df


def _build_insights(
    *,
    total_spend: float,
    delta_label: str,
    projection: ProjectionResult,
    confidence: float,
    top_category: Optional[pd.Series],
    top_vendor: Optional[VendorRow],
) -> list[str]:
    insights: list[str] = []

    conf_pct = int(round(confidence * 100))
    if abs(projection.high - projection.low) < 1e-6:
        projection_text = f"Projection: <strong>£{projection.projected_total:,.0f}</strong>"
    else:
        projection_text = (
            f"Projection: <strong>£{projection.low:,.0f}–£{projection.high:,.0f}</strong>"
        )

    insights.append(
        (
            f"You've spent <strong>£{total_spend:,.0f}</strong> so far ({delta_label}). "
            f"{projection_text} ({conf_pct}% confidence)."
        )
    )

    if top_category is not None and not top_category.empty:
        insights.append(
            (
                f"Top category: <strong>{top_category['Category']}</strong> at "
                f"£{top_category['CurrentValue']:,.0f}."
            )
        )

    if top_vendor is not None:
        insights.append(
            (
                f"Largest merchant: <strong>{top_vendor['label']}</strong> "
                f"(£{top_vendor['amount']:,.0f})."
            )
        )

    return insights


def _format_delta(current: float, previous: float) -> str:
    if previous <= 0:
        if current <= 0:
            return "No change vs last month"
        return "New vs last month"

    change = (current - previous) / previous
    sign = "+" if change >= 0 else ""
    return f"{sign}{change * 100:.1f}% vs last month"

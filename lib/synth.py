
"""Synthetic UK debit transaction generator for the spending analyser.

Produces realistic bank statement style data for development and testing.
The generator focuses on United Kingdom merchants, sensible spending cadences,
and a column set commonly seen in debit account exports.
"""

from __future__ import annotations

import calendar
import itertools
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


FIELDS: Tuple[str, ...] = (
    "txn_id",
    "date",
    "amount",
    "currency",
    "description",
    "mcc",
    "category",
    "channel",
    "merchant_city",
    "merchant_country",
    "card_last4",
    "is_refund",
    "note",
)

ALLOWED_CHANNELS: Tuple[str, ...] = (
    "card_present",
    "online",
    "direct_debit",
    "standing_order",
    "bank_transfer",
)


@dataclass(frozen=True)
class MerchantProfile:
    """Metadata describing a merchant used in synthetic ledgers."""

    description: str
    mcc: str
    category: str
    channel: str
    city: str
    country: str
    default_note: str = ""


INCOME_MERCHANTS: Sequence[MerchantProfile] = (
    MerchantProfile(
        description="BACS CREDIT THAMES TECH LTD",
        mcc="6011",
        category="income",
        channel="bank_transfer",
        city="London",
        country="GBR",
        default_note="Monthly salary payment",
    ),
    MerchantProfile(
        description="BACS CREDIT FREELANCE HUB",
        mcc="6011",
        category="income",
        channel="bank_transfer",
        city="Manchester",
        country="GBR",
        default_note="Freelance services payout",
    ),
)

RENT_MERCHANT = MerchantProfile(
    description="STANDING ORDER RENT OAKWOOD ESTATES",
    mcc="6513",
    category="rent",
    channel="standing_order",
    city="London",
    country="GBR",
    default_note="Monthly rent",
)

UTILITY_MERCHANTS: Sequence[Tuple[MerchantProfile, float, float]] = (
    (
        MerchantProfile(
            description="DIRECT DEBIT OCTOPUS ENERGY",
            mcc="4900",
            category="utilities",
            channel="direct_debit",
            city="London",
            country="GBR",
            default_note="Electricity bill",
        ),
        -120.0,
        0.07,
    ),
    (
        MerchantProfile(
            description="DIRECT DEBIT THAMES WATER",
            mcc="4900",
            category="utilities",
            channel="direct_debit",
            city="Reading",
            country="GBR",
            default_note="Water services",
        ),
        -48.0,
        0.05,
    ),
    (
        MerchantProfile(
            description="DIRECT DEBIT VIRGIN MEDIA",
            mcc="4899",
            category="utilities",
            channel="direct_debit",
            city="Manchester",
            country="GBR",
            default_note="Broadband subscription",
        ),
        -56.0,
        0.06,
    ),
)

GROCERY_MERCHANTS: Sequence[MerchantProfile] = (
    MerchantProfile(
        description="TESCO EXPRESS LONDON",
        mcc="5411",
        category="groceries",
        channel="card_present",
        city="London",
        country="GBR",
    ),
    MerchantProfile(
        description="SAINSBURYS LOCAL",
        mcc="5411",
        category="groceries",
        channel="card_present",
        city="London",
        country="GBR",
    ),
    MerchantProfile(
        description="WAITROSE & PARTNERS",
        mcc="5411",
        category="groceries",
        channel="card_present",
        city="Guildford",
        country="GBR",
    ),
    MerchantProfile(
        description="OCADO RETAIL ONLINE",
        mcc="5499",
        category="groceries",
        channel="online",
        city="Hatfield",
        country="GBR",
    ),
)

TRANSPORT_MERCHANTS: Sequence[MerchantProfile] = (
    MerchantProfile(
        description="TFL TRAVEL CHARGE",
        mcc="4111",
        category="transport",
        channel="card_present",
        city="London",
        country="GBR",
    ),
    MerchantProfile(
        description="UBER TRIP HELP.UBER.COM",
        mcc="4121",
        category="transport",
        channel="online",
        city="London",
        country="GBR",
    ),
    MerchantProfile(
        description="TRAINLINE TICKETS",
        mcc="4112",
        category="transport",
        channel="online",
        city="Edinburgh",
        country="GBR",
    ),
)

SUBSCRIPTION_MERCHANTS: Sequence[Tuple[MerchantProfile, float]] = (
    (
        MerchantProfile(
            description="DIRECT DEBIT NETFLIX.COM",
            mcc="4899",
            category="subscriptions",
            channel="direct_debit",
            city="London",
            country="GBR",
            default_note="Streaming subscription",
        ),
        -11.99,
    ),
    (
        MerchantProfile(
            description="DIRECT DEBIT SPOTIFY UK",
            mcc="5815",
            category="subscriptions",
            channel="direct_debit",
            city="London",
            country="GBR",
            default_note="Music subscription",
        ),
        -9.99,
    ),
    (
        MerchantProfile(
            description="DIRECT DEBIT PUREGYM",
            mcc="7991",
            category="subscriptions",
            channel="direct_debit",
            city="Leeds",
            country="GBR",
            default_note="Gym membership",
        ),
        -24.99,
    ),
)

EVENT_MERCHANTS: Sequence[Tuple[MerchantProfile, Tuple[float, float]]] = (
    (
        MerchantProfile(
            description="O2 ARENA LONDON",
            mcc="7922",
            category="entertainment",
            channel="card_present",
            city="London",
            country="GBR",
        ),
        (-240.0, -80.0),
    ),
    (
        MerchantProfile(
            description="PREMIER LEAGUE TICKETS",
            mcc="7941",
            category="entertainment",
            channel="card_present",
            city="Manchester",
            country="GBR",
        ),
        (-180.0, -60.0),
    ),
    (
        MerchantProfile(
            description="THEATRE ROYAL DRURY",
            mcc="7922",
            category="entertainment",
            channel="card_present",
            city="London",
            country="GBR",
        ),
        (-150.0, -70.0),
    ),
)

CARD_LAST4_POOL: Sequence[str] = ("9812", "4456", "2201", "0065")


def generate_synthetic_transactions(
    start_date: date | datetime | str | None = None,
    months: Optional[int] = None,
    *,
    end_date: date | datetime | str | None = None,
    months_full: int = 6,
    include_current_partial: bool = True,
    currency: str = "GBP",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a synthetic ledger of UK debit transactions.

    Two operating modes are supported:

    1. Provide ``start_date`` and ``months`` to generate a fixed number of
       complete months from the chosen starting point.
    2. Omit ``start_date`` to automatically build the last ``months_full``
       complete months plus the current month up to ``end_date`` (today by
       default). This matches typical "recent statement" exports.
    """

    rng = np.random.default_rng(seed)

    today = date.today()
    end_date_obj = _normalize_date(end_date) if end_date is not None else today
    end_date_obj = min(end_date_obj, today)

    if start_date is not None:
        start = _normalize_date(start_date)
        if months is None:
            raise ValueError("`months` must be provided when `start_date` is set")
        if months <= 0:
            raise ValueError("months must be a positive integer")
        month_starts = [_add_months(start, offset) for offset in range(months)]
        period_start = start
        period_end = _add_months(start, months) - timedelta(days=1)
    else:
        if months_full <= 0:
            raise ValueError("months_full must be a positive integer")
        current_month_start = _month_floor(end_date_obj)
        period_start = _add_months(current_month_start, -months_full)
        if include_current_partial:
            period_end = end_date_obj
        else:
            period_end = current_month_start - timedelta(days=1)

        month_starts: List[date] = []
        anchor = period_start
        final_month_start = _month_floor(period_end)
        while anchor <= final_month_start:
            month_starts.append(anchor)
            anchor = _add_months(anchor, 1)

    if period_start > period_end:
        return pd.DataFrame(columns=FIELDS)

    all_days = pd.date_range(start=period_start, end=period_end, freq="D")
    week_keys = sorted({(d.isocalendar().year, d.isocalendar().week) for d in all_days})
    num_event_weeks = min(max(1, len(month_starts) // 2), len(week_keys)) if week_keys else 0
    event_weeks = set()
    if num_event_weeks:
        event_week_indices = rng.choice(len(week_keys), size=num_event_weeks, replace=False)
        event_weeks = {week_keys[int(idx)] for idx in np.atleast_1d(event_week_indices)}

    records: List[dict] = []
    txn_counter = itertools.count(1)

    def append_record(
        txn_date: date,
        amount: float,
        merchant: MerchantProfile,
        *,
        note: Optional[str] = None,
        is_refund: bool = False,
        card_last4: Optional[str] = None,
    ) -> None:
        if txn_date < period_start or txn_date > period_end:
            return
        if merchant.channel not in ALLOWED_CHANNELS:
            raise ValueError(f"Unsupported channel '{merchant.channel}' for {merchant.description}")

        record_note = note or merchant.default_note
        amount_value = -amount if is_refund and amount < 0 else amount

        records.append(
            {
                "txn_id": f"txn_{next(txn_counter):06d}",
                "date": txn_date.isoformat(),
                "amount": round(amount_value, 2),
                "currency": currency,
                "description": merchant.description,
                "mcc": merchant.mcc,
                "category": merchant.category,
                "channel": merchant.channel,
                "merchant_city": merchant.city,
                "merchant_country": merchant.country,
                "card_last4": card_last4 or _rng_choice(CARD_LAST4_POOL, rng),
                "is_refund": bool(is_refund),
                "note": record_note,
            }
        )

    for month_anchor in month_starts:
        year, month = month_anchor.year, month_anchor.month

        # Income on 1st and 15th where applicable
        for payday_day in (1, 15):
            payday_date = _clamp_day(year, month, payday_day)
            employer = _rng_choice(INCOME_MERCHANTS, rng)
            base_income = rng.normal(2650, 140)
            bonus = rng.normal(420, 70) if payday_day == 15 and rng.random() < 0.22 else 0
            amount = abs(base_income + bonus)
            append_record(payday_date, amount, employer, card_last4="0000")

        # Rent at first weekday of the month
        rent_date = _first_weekday(year, month)
        rent_amount = -abs(rng.normal(1780, 60))
        append_record(rent_date, rent_amount, RENT_MERCHANT)

        # Utilities
        utility_days = (6, 12, 20)
        for (merchant, base_amount, drift_pct), utility_day in zip(UTILITY_MERCHANTS, utility_days):
            utility_date = _clamp_day(year, month, utility_day)
            amount = base_amount * (1 + rng.normal(0, drift_pct))
            append_record(utility_date, amount, merchant)

        # Subscriptions
        subscription_day = 8
        for merchant, base_amount in SUBSCRIPTION_MERCHANTS:
            offset = int(rng.integers(-2, 3))
            sub_date = _clamp_day(year, month, subscription_day + offset)
            drift = base_amount * rng.normal(0, 0.015)
            append_record(sub_date, base_amount + drift, merchant)

        month_dates_full = _month_date_range(year, month)
        month_dates = [d for d in month_dates_full if period_start <= d <= period_end]
        if not month_dates:
            continue

        # Groceries with event week variation
        grocery_count = int(rng.integers(6, 9))
        for _ in range(grocery_count):
            txn_date = _rng_choice(month_dates, rng)
            week_key = _week_key(txn_date)
            multiplier = 1.35 if week_key in event_weeks else 1.0
            amount = -abs(rng.normal(62, 18) * multiplier)
            merchant = _rng_choice(GROCERY_MERCHANTS, rng)
            append_record(txn_date, amount, merchant, note="Groceries")

        # Transport
        transport_count = int(rng.integers(10, 18))
        for _ in range(transport_count):
            txn_date = _rng_choice(month_dates, rng)
            week_key = _week_key(txn_date)
            multiplier = 1.4 if week_key in event_weeks else 1.0
            amount = -abs(rng.uniform(2.8, 28.0) * multiplier)
            merchant = _rng_choice(TRANSPORT_MERCHANTS, rng)
            note = "Commute" if merchant.description.startswith("TFL") else "Travel"
            append_record(txn_date, amount, merchant, note=note)

        # Event week spending
        for week_key in sorted(event_weeks):
            if _is_week_in_month(week_key, year, month):
                week_start = _week_start_date(week_key)
                event_date = week_start + timedelta(days=int(rng.integers(0, 7)))
                merchant, amount_bounds = _rng_choice(EVENT_MERCHANTS, rng)
                amount = rng.uniform(*amount_bounds)
                append_record(event_date, amount, merchant, note="Event week treat")

    _inject_refunds(records, rng)
    _inject_duplicates(records, rng)

    df = pd.DataFrame.from_records(records, columns=FIELDS)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def write_transactions_csv(
    path: str,
    *,
    seed: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Generate synthetic data and persist it to ``path``.

    Additional keyword arguments are forwarded to
    :func:`generate_synthetic_transactions`.
    """

    df = generate_synthetic_transactions(seed=seed, **kwargs)
    df.to_csv(path, index=False)
    return df


def _normalize_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed.date()
    raise TypeError(f"Unsupported date value: {value!r}")


def _add_months(anchor: date, months: int) -> date:
    month = anchor.month - 1 + months
    year = anchor.year + month // 12
    month = month % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _month_floor(moment: date) -> date:
    return moment.replace(day=1)


def _clamp_day(year: int, month: int, day: int) -> date:
    _, max_day = calendar.monthrange(year, month)
    return date(year, month, max(1, min(day, max_day)))


def _first_weekday(year: int, month: int) -> date:
    candidate = date(year, month, 1)
    while candidate.weekday() >= 5:  # 0=Mon, 5=Sat, 6=Sun
        candidate += timedelta(days=1)
    return candidate


def _month_date_range(year: int, month: int) -> List[date]:
    start = date(year, month, 1)
    end = _add_months(start, 1) - timedelta(days=1)
    return [d.date() for d in pd.date_range(start=start, end=end, freq="D")]


def _week_key(moment: date) -> Tuple[int, int]:
    iso_year, iso_week, _ = moment.isocalendar()
    return iso_year, iso_week


def _week_start_date(week_key: Tuple[int, int]) -> date:
    iso_year, iso_week = week_key
    return date.fromisocalendar(iso_year, iso_week, 1)


def _is_week_in_month(week_key: Tuple[int, int], year: int, month: int) -> bool:
    week_start = _week_start_date(week_key)
    week_end = week_start + timedelta(days=6)
    return week_start.month == month or week_end.month == month


def _rng_choice(options: Sequence[T], rng: np.random.Generator) -> T:
    if not options:
        raise ValueError("Cannot choose from an empty sequence")
    idx = int(rng.integers(0, len(options)))
    return options[idx]


def _inject_duplicates(records: List[dict], rng: np.random.Generator, fraction: float = 0.02) -> None:
    if not records:
        return
    dup_count = max(1, int(len(records) * fraction))
    indices = rng.choice(len(records), size=dup_count, replace=False)
    for idx in indices:
        original = records[idx]
        duplicate = original.copy()
        duplicate["txn_id"] = f"{original['txn_id']}_dup"
        duplicate["note"] = (original.get("note") or "").strip()
        duplicate["note"] = (duplicate["note"] + " | potential duplicate").strip()
        records.append(duplicate)


def _inject_refunds(records: List[dict], rng: np.random.Generator, probability: float = 0.02) -> None:
    for record in records:
        if record["amount"] < 0 and rng.random() < probability:
            record["amount"] = abs(record["amount"])
            record["is_refund"] = True
            note = (record.get("note") or "").strip()
            record["note"] = (note + " | refund").strip()
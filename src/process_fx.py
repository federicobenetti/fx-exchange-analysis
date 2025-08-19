#!/usr/bin/env python3
"""
process_fx.py
Utilities to load, clean, and engineer features for a daily FX dataset.

Expected columns: currency, base_currency, currency_name, exchange_rate, date
Assumption: exchange_rate = units of base_currency per 1 unit of `currency`.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Tuple

def load_fx(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize columns
    rename = {
        "currency": "currency",
        "base_currency": "base_currency",
        "currency_name": "currency_name",
        "exchange_rate": "exchange_rate",
        "date": "date"
    }
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=rename)
    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False).dt.tz_localize(None)
    # Ensure dtypes
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df["base_currency"] = df["base_currency"].astype(str).str.upper().str.strip()
    # Numeric exchange rate
    df["exchange_rate"] = pd.to_numeric(df["exchange_rate"], errors="coerce")
    # Drop full-NA rows
    df = df.dropna(subset=["currency", "base_currency", "exchange_rate", "date"])
    # Sort
    df = df.sort_values(["currency","base_currency","date"]).reset_index(drop=True)
    # Drop dups (keep last)
    df = df.drop_duplicates(subset=["currency","base_currency","date"], keep="last")
    return df

def subset_pair(df: pd.DataFrame, base: str, curr: str) -> pd.DataFrame:
    mask = (df["base_currency"] == base.upper()) & (df["currency"] == curr.upper())
    out = df.loc[mask, ["date","exchange_rate"]].sort_values("date").reset_index(drop=True)
    out = out.set_index("date").asfreq("D")
    # Forward-fill small gaps (market closed days), limit to 7 days to be safe
    out["exchange_rate"] = out["exchange_rate"].ffill(limit=7)
    return out

def compute_returns(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": series})
    df["ret"] = df["price"].pct_change()
    df["log_ret"] = np.log(df["price"]).diff()
    return df

def rolling_vol(log_ret: pd.Series, window: int = 30) -> pd.Series:
    # Annualize by sqrt(252) after std of daily log returns
    return log_ret.rolling(window).std() * np.sqrt(252)

def cross_rate(df: pd.DataFrame, base: str, from_ccy: str, to_ccy: str, on_date: str, ffill_days: int = 7) -> float:
    """Compute cross-rate (to_ccy per 1 from_ccy) using base. Example: EUR->JPY using base USD."""
    base = base.upper(); f = from_ccy.upper(); t = to_ccy.upper()
    df_base = df[(df["base_currency"] == base)]
    # get base_per_from and base_per_to
    a = df_base[df_base["currency"] == f][["date","exchange_rate"]].set_index("date").asfreq("D").ffill(limit=ffill_days)
    b = df_base[df_base["currency"] == t][["date","exchange_rate"]].set_index("date").asfreq("D").ffill(limit=ffill_days)
    date = pd.to_datetime(on_date)
    try:
        rate = float(a.loc[date, "exchange_rate"] / b.loc[date, "exchange_rate"])
    except KeyError:
        raise KeyError(f"No rate available for date {on_date}. Try a nearby business day or increase ffill_days.")
    return rate

def wide_index(df: pd.DataFrame, base: str, currencies: Iterable[str], start_index: float = 100.0) -> pd.DataFrame:
    """Return a DataFrame with normalized index (start at 100) per currency vs base."""
    base = base.upper()
    currencies = [c.upper() for c in currencies]
    frames = []
    for c in currencies:
        s = subset_pair(df, base, c)["exchange_rate"].rename(c)
        s = (s / s.dropna().iloc[0]) * start_index
        frames.append(s)
    out = pd.concat(frames, axis=1)
    return out

def ensure_monotonic_dates(s: pd.Series) -> pd.Series:
    # Deduplicate if same-day duplicates exist (take last)
    return s[~s.index.duplicated(keep="last")]

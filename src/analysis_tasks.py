#!/usr/bin/env python3
"""
analysis_tasks.py
CLI for FX analytics tasks: trend, forecast, compare, volatility, convert, event_impact.
"""

from __future__ import annotations
import argparse, sys, os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from typing import List
from process_fx import load_fx, subset_pair, compute_returns, rolling_vol, cross_rate, wide_index

def save_plot(fig, output: str):
    # Save static PNG if kaleido available, else save HTML
    try:
        fig.write_image(output)
        print(f"Saved figure to {output}")
    except Exception as e:
        html_out = os.path.splitext(output)[0] + ".html"
        fig.write_html(html_out)
        print(f"Could not save PNG ({e}); saved HTML instead at {html_out}")

def cmd_trend(args):
    df = load_fx(args.input)
    pair = subset_pair(df, args.base, args.currency)
    pair = pair.dropna()
    ser = pair["exchange_rate"]
    # Moving averages
    ma_short = ser.rolling(args.ma_short).mean()
    ma_long = ser.rolling(args.ma_long).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ser.index, y=ser, name=f"{args.currency}/{args.base}", mode="lines"))
    fig.add_trace(go.Scatter(x=ma_short.index, y=ma_short, name=f"MA{args.ma_short}", mode="lines"))
    fig.add_trace(go.Scatter(x=ma_long.index, y=ma_long, name=f"MA{args.ma_long}", mode="lines"))
    fig.update_layout(title=f"Trend: {args.currency}/{args.base}", xaxis_title="Date", yaxis_title=f"{args.base} per 1 {args.currency}")
    save_plot(fig, args.output)

def cmd_forecast(args):
    df = load_fx(args.input)
    pair = subset_pair(df, args.base, args.currency).dropna()
    y = pair["exchange_rate"].asfreq("D")
    y = y.fillna(method="ffill", limit=7).dropna()
    steps = args.steps

    if args.model == "ets":
        model = ExponentialSmoothing(y, trend="add", seasonal=None, damped_trend=True)
        fit = model.fit(optimized=True, use_brute=True)
        forecast = fit.forecast(steps)
    elif args.model == "arima":
        model = ARIMA(y, order=(1,1,1))
        fit = model.fit()
        forecast = fit.forecast(steps)
    else:
        raise ValueError("Unknown model; use 'ets' or 'arima'")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, name="History"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name=f"Forecast ({args.model.upper()})"))
    fig.update_layout(title=f"Forecast: {args.currency}/{args.base} (next {steps} days)", xaxis_title="Date", yaxis_title=f"{args.base} per 1 {args.currency}")
    save_plot(fig, args.output)

def cmd_compare(args):
    df = load_fx(args.input)
    currencies = [c.strip() for c in args.currencies.split(",")]
    idx = wide_index(df, args.base, currencies, start_index=100.0)
    fig = px.line(idx, x=idx.index, y=idx.columns, labels={"value":"Index (start=100)", "variable":"Currency"}, title=f"Normalized Index vs {args.base}")
    save_plot(fig, args.output)

def cmd_volatility(args):
    df = load_fx(args.input)
    currencies = [c.strip() for c in args.currencies.split(",")]
    vols = {}
    for c in currencies:
        ser = subset_pair(df, args.base, c)["exchange_rate"].dropna()
        ret = np.log(ser).diff()
        vol = ret.rolling(args.window).std() * np.sqrt(252)
        vols[c] = vol
    vol_df = pd.DataFrame(vols)
    fig = px.line(vol_df, x=vol_df.index, y=vol_df.columns, labels={"value":"Annualized Volatility", "variable":"Currency"}, title=f"{args.window}D Rolling Volatility vs {args.base}")
    save_plot(fig, args.output)

def cmd_convert(args):
    df = load_fx(args.input)
    rate = cross_rate(df, args.base, args.from_currency, args.to_currency, args.date, ffill_days=args.ffill_days)
    converted = args.amount * rate
    print(f"{args.amount:.4f} {args.from_currency.upper()} = {converted:.4f} {args.to_currency.upper()} on {args.date} (via base {args.base.upper()}, cross-rate {rate:.6f} {args.to_currency.upper()}/{args.from_currency.upper()})")

def cmd_event_impact(args):
    df = load_fx(args.input)
    pair = subset_pair(df, args.base, args.currency)["exchange_rate"].dropna()
    ret = np.log(pair).diff().dropna()
    events = pd.read_csv(args.events)
    events["date"] = pd.to_datetime(events["date"])

    rows = []
    win = args.window
    for _, r in events.iterrows():
        d = pd.to_datetime(r["date"])
        w = ret.loc[d - pd.Timedelta(days=win): d + pd.Timedelta(days=win)]
        if w.empty:
            continue
        # Cumulative return over window
        cum = w.sum()
        rows.append({"event_date": d.date(), "event": r.get("event", ""), "cum_log_return": float(cum)})

    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("No matching dates found for events within the data range.")
    else:
        out.to_csv(args.output, index=False)
        print(f"Saved event impact table to {args.output}")
        print(out.head())

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("trend", help="Trend analysis with moving averages")
    p1.add_argument("--input", required=True)
    p1.add_argument("--base", required=True)
    p1.add_argument("--currency", required=True)
    p1.add_argument("--ma_short", type=int, default=20)
    p1.add_argument("--ma_long", type=int, default=60)
    p1.add_argument("--output", required=True)
    p1.set_defaults(func=cmd_trend)

    p2 = sp.add_parser("forecast", help="Forecast using ETS or ARIMA")
    p2.add_argument("--input", required=True)
    p2.add_argument("--base", required=True)
    p2.add_argument("--currency", required=True)
    p2.add_argument("--steps", type=int, default=14)
    p2.add_argument("--model", choices=["ets","arima"], default="ets")
    p2.add_argument("--output", required=True)
    p2.set_defaults(func=cmd_forecast)

    p3 = sp.add_parser("compare", help="Compare normalized index across currencies")
    p3.add_argument("--input", required=True)
    p3.add_argument("--base", required=True)
    p3.add_argument("--currencies", required=True, help="Comma-separated list, e.g., EUR,GBP,JPY")
    p3.add_argument("--output", required=True)
    p3.set_defaults(func=cmd_compare)

    p4 = sp.add_parser("volatility", help="Rolling volatility")
    p4.add_argument("--input", required=True)
    p4.add_argument("--base", required=True)
    p4.add_argument("--currencies", required=True)
    p4.add_argument("--window", type=int, default=30)
    p4.add_argument("--output", required=True)
    p4.set_defaults(func=cmd_volatility)

    p5 = sp.add_parser("convert", help="Currency conversion on a given date using cross-rates")
    p5.add_argument("--input", required=True)
    p5.add_argument("--from_currency", required=True)
    p5.add_argument("--to_currency", required=True)
    p5.add_argument("--base", required=True)
    p5.add_argument("--date", required=True)
    p5.add_argument("--amount", type=float, required=True)
    p5.add_argument("--ffill_days", type=int, default=7)
    p5.set_defaults(func=cmd_convert)

    p6 = sp.add_parser("event_impact", help="Event window analysis (±window days) using log returns")
    p6.add_argument("--input", required=True)
    p6.add_argument("--base", required=True)
    p6.add_argument("--currency", required=True)
    p6.add_argument("--events", required=True, help="CSV with columns: date,event")
    p6.add_argument("--window", type=int, default=5)
    p6.add_argument("--output", required=True)
    p6.set_defaults(func=cmd_event_impact)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

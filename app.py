import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.process_fx import load_fx, subset_pair, compute_returns, rolling_vol, cross_rate, wide_index

st.set_page_config(page_title="FX Exchange Analytics", layout="wide")
st.title("💱 FX Exchange Analytics")

st.write("Upload your daily FX dataset or place it in `data/fx_daily.csv`. "
         "Select base/currencies to explore trends, compare, check volatility, convert, and forecast.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def load_data(file_or_path):
    if file_or_path is None:
        return None
    if isinstance(file_or_path, str):
        return load_fx(file_or_path)
    else:
        # file-like from uploader
        df = pd.read_csv(file_or_path)
        tmp = "/tmp/_fx_uploaded.csv"
        df.to_csv(tmp, index=False)
        return load_fx(tmp)

# Try local default if exists
import os
default_path = "data/fx_daily.csv"
df = None
if uploaded is not None:
    df = load_data(uploaded)
elif os.path.exists(default_path):
    df = load_data(default_path)

if df is None:
    st.info("Upload a CSV or place one at `data/fx_daily.csv` to begin.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Filters")
    base_options = sorted(df["base_currency"].unique().tolist())
    base = st.selectbox("Base currency", base_options, index=base_options.index("USD") if "USD" in base_options else 0)

    # Currencies available against base
    ccy_list = sorted(df[df["base_currency"] == base]["currency"].unique().tolist())
    ccy = st.selectbox("Currency", ccy_list, index=ccy_list.index("EUR") if "EUR" in ccy_list else 0)

    compare_list = st.multiselect("Compare currencies", ccy_list, default=[c for c in ["EUR","GBP","JPY"] if c in ccy_list])

    date_min = pd.to_datetime(df["date"]).min().date()
    date_max = pd.to_datetime(df["date"]).max().date()
    date_range = st.date_input("Date range", value=(date_min, date_max))

# Filter pair for main panels
pair = subset_pair(df, base, ccy).dropna()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = map(pd.to_datetime, date_range)
    pair = pair.loc[start:end]

# KPIs
st.subheader(f"KPI — {ccy}/{base}")
ser = pair["exchange_rate"]
ret = np.log(ser).diff()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest rate", f"{ser.dropna().iloc[-1]:.6f} {base}/{ccy}")
col2.metric("Daily change", f"{(ret.iloc[-1]*100):.3f}%" if not ret.empty and not np.isnan(ret.iloc[-1]) else "N/A")
col3.metric("30D vol (ann.)", f"{(ret.rolling(30).std().iloc[-1]*np.sqrt(252)*100):.2f}%" if len(ret)>=30 else "N/A")
col4.metric("YTD change", f"{((ser/ser[ser.index.year==ser.index[-1].year].iloc[0]-1).iloc[-1]*100):.2f}%" if not ser.empty else "N/A")

st.divider()

# Charts
c1, c2 = st.columns(2)
with c1:
    st.subheader(f"Price — {ccy}/{base}")
    fig = px.line(pair, x=pair.index, y="exchange_rate", labels={"exchange_rate": f"{base}/{ccy}"})
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Rolling Volatility (30D, annualized)")
    vol = ret.rolling(30).std() * np.sqrt(252)
    fig2 = px.line(vol, x=vol.index, y=vol.values, labels={"x":"Date","y":"Ann. Vol"})
    st.plotly_chart(fig2, use_container_width=True)

# Compare index
if compare_list:
    st.subheader(f"Comparison vs {base} (Index=100 at start)")
    idx = wide_index(df, base, compare_list, start_index=100.0)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        idx = idx.loc[pd.to_datetime(date_range[0]):pd.to_datetime(date_range[1])]
    fig3 = px.line(idx, x=idx.index, y=idx.columns, labels={"value":"Index","variable":"Currency"})
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Conversion tool
st.subheader("Conversion")
c_from, c_to, c_amt, c_date = st.columns([1,1,1,1])
with c_from:
    from_ccy = st.selectbox("From", ccy_list, index=ccy_list.index("EUR") if "EUR" in ccy_list else 0, key="fromcur")
with c_to:
    to_ccy = st.selectbox("To", ccy_list, index=ccy_list.index("USD") if "USD" in ccy_list else 0, key="tocur")
with c_amt:
    amount = st.number_input("Amount", value=100.0, min_value=0.0, step=10.0)
with c_date:
    d = st.date_input("Date", value=date_max)

if st.button("Convert"):
    try:
        rate = cross_rate(df, base, from_ccy, to_ccy, str(d))
        st.success(f"{amount:.2f} {from_ccy} = {amount*rate:.2f} {to_ccy} on {d} (via base {base}; rate {rate:.6f} {to_ccy}/{from_ccy})")
    except Exception as e:
        st.error(str(e))

st.divider()

# Forecast (quick baseline)
st.subheader("Forecast (Baseline)")
horizon = st.slider("Steps (days)", 7, 60, 14)
model_choice = st.selectbox("Model", ["ETS (Holt-Winters)", "ARIMA (1,1,1)"])

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA

    y = subset_pair(df, base, ccy)["exchange_rate"].dropna().asfreq("D").ffill(limit=7)

    if model_choice.startswith("ETS"):
        model = ExponentialSmoothing(y, trend="add", seasonal=None, damped_trend=True)
        fit = model.fit(optimized=True, use_brute=True)
        fc = fit.forecast(horizon)
    else:
        model = ARIMA(y, order=(1,1,1))
        fit = model.fit()
        fc = fit.forecast(horizon)

    figf = go.Figure()
    figf.add_trace(go.Scatter(x=y.index, y=y, name="History"))
    figf.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast"))
    st.plotly_chart(figf, use_container_width=True)
except Exception as e:
    st.warning(f"Forecast unavailable: {e}")

# Data preview
st.subheader("Data Preview")
st.dataframe(df.head(50))

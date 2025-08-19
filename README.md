# FX Exchange Analytics 📈💱

Clean, analyze, and visualize **daily FX exchange rates** (160+ currencies) with ready-made scripts and a Streamlit dashboard.

**Dataset columns**: `currency`, `base_currency`, `currency_name`, `exchange_rate`, `date`  
> Assumption: `exchange_rate` = units of **base_currency per 1 unit of `currency`** (e.g., if base = USD and currency = EUR, 1 EUR = 1.07 USD).

---

## 🔧 What’s inside
- `src/process_fx.py` → Load, clean, and feature engineer FX data (returns, log-returns, rolling volatility).
- `src/analysis_tasks.py` → CLI with tasks:
  - **trend**: moving averages & patterns
  - **forecast**: simple ETS (Holt-Winters) or ARIMA baseline
  - **compare**: index normalization across currencies
  - **volatility**: rolling volatility by currency
  - **convert**: currency conversion for a given date
  - **event_impact**: event-window analysis on returns
- `app.py` → Streamlit dashboard (filter, compare, volatility, conversion, forecast)
- `requirements.txt`, `.gitignore`, `data/events_template.csv`

---

## 📁 Repo Structure
```
fx-exchange-analytics/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── fx_daily.csv                # put your dataset here (excluded from git by default)
│   └── events_template.csv         # sample structure for event analysis
├── results/                        # exported charts/tables
└── src/
    ├── process_fx.py
    └── analysis_tasks.py
```

---

## 🚀 Quickstart
Create a virtual environment (optional) and install dependencies:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

**Place your CSV** in `data/fx_daily.csv` (or upload via Streamlit).

### Run tasks from CLI
```bash
# 1) Trend analysis (EUR vs USD)
python src/analysis_tasks.py trend --input data/fx_daily.csv --base USD --currency EUR --output results/eur_trend.png

# 2) Forecast next 14 days with ETS
python src/analysis_tasks.py forecast --input data/fx_daily.csv --base USD --currency EUR --steps 14 --model ets --output results/eur_forecast.png

# 3) Compare normalized index across multiple currencies
python src/analysis_tasks.py compare --input data/fx_daily.csv --base USD --currencies EUR,GBP,JPY --output results/compare_index.png

# 4) Volatility (30d rolling)
python src/analysis_tasks.py volatility --input data/fx_daily.csv --base USD --currencies EUR,GBP,JPY --window 30 --output results/volatility.png

# 5) Convert 100 EUR → JPY on 2024-03-15
python src/analysis_tasks.py convert --input data/fx_daily.csv --amount 100 --from_currency EUR --to_currency JPY --base USD --date 2024-03-15

# 6) Event impact around CPI/Fed dates (±5 days window)
python src/analysis_tasks.py event_impact --input data/fx_daily.csv --base USD --currency EUR --events data/events_template.csv --window 5 --output results/event_impact.csv
```

### Run the dashboard
```bash
streamlit run app.py
```
Then open the URL shown in terminal.

---

## 🧠 Notes & Assumptions
- Data types are inferred and `date` is parsed to datetime.
- If there are multiple `base_currency` values, you can **filter** by base in CLI or Streamlit.
- Cross-rate conversions use: `(base_per_from) / (base_per_to)`.
- Forecasts are simple baselines for demonstration; always validate before using for decisions.

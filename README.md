# PlainSpend Dashboard

A Streamlit dashboard for exploring synthetic spending insights, featuring polished card-based layouts and Altair visualisations.

## Project structure

- `app.py` — Streamlit entrypoint orchestrating layout and data loading.
- `lib/` — Data preparation utilities and forecasting helpers.
- `ui/components.py` — Reusable UI building blocks (cards, global styles).
- `ui/charts.py` — Centralised Altair chart builders for consistency across the app.
- `data/seed.csv` — Seed dataset used to power the dashboard.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

## Development tips

- Chart styling lives in `ui/charts.py`; update there to keep visuals consistent.
- Cached data loading follows Streamlit best practices via `@st.cache_data` to avoid redundant CSV reads.
- UI tweaks such as card layout or navigation styles belong in `ui/components.py`.

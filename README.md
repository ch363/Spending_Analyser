# PlainSpend Dashboard

A Streamlit dashboard for exploring synthetic spending insights, featuring polished card-based layouts and interactive Plotly visualisations.

## Project structure

- `app/main.py` — Streamlit entrypoint orchestrating layout, routing, and theming.
- `app/layout.py` & `app/pages/` — Reusable layout primitives and Streamlit page compositions.
- `analytics/` — Core analytics helpers (categorisation, recurring detection, forecasting, summary calculations).
- `core/summary_service.py` — Dashboard aggregation pipeline that orchestrates analytics outputs.
- `core/summary.py` — High-level summary + AI integration layer (bullet generation, OpenAI client wiring).
- `core/data_loader.py` — Cached CSV reader powering the analytics stack.
- `core/models.py` — Typed data structures shared across the dashboard.
- `config/settings.py` — Centralised Pydantic settings for OpenAI credentials and defaults.
- `visualization/charts.py` — Centralised Plotly chart builders for consistency across the app.
- `prompts/` — Prompt templates consumed by AI features.
- `data/seed.csv` — Seed dataset used to power the dashboard.
- `tests/` — Pytest suite covering analytics helpers and end-to-end data prep.

## Getting started

1. Install dependencies (inside a virtual environment if you prefer isolation):

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Configure your OpenAI credentials:

   - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`.
   - Update the `api_key` value with your OpenAI key and, if needed, override the model or base URL.
   - Keep the real `secrets.toml` out of version control.

3. Launch the dashboard:

   ```bash
   streamlit run app/main.py
   ```
### Synthetic data
The app uses a synthetic dataset generated from `data/synth.py`. You can modify parameters like date range, income, and spending patterns by editing the script and re-running it to produce a new `data/seed.csv`.
Update the `data/seed.csv` file by running:

```bash
python3 data/synth.py

./.venv/bin/python3 data/synth.py
```

### Running tests

With dependencies installed, run the suite from the repo root:

```bash
PYTHONPATH=. pytest
```

### AI client overrides

During tests you can inject a stand-in OpenAI client via dependency injection hooks on `core.summary.generate_ai_summary`, avoiding real API calls.

## Pages

- **Overview** — The main dashboard with monthly metrics, charts, and AI highlights.
- **Insights** — An AI-guided deep dive featuring the cash-flow runway gauge, recurring subscription watchlist, duplicate charge alerts, and top category movers. Access it from the navigation bar at the top of the app.

## Development tips

- Chart styling lives in `visualization/charts.py`; update there to keep visuals consistent.
- Cached data loading follows Streamlit best practices via `@st.cache_data` to avoid redundant CSV reads.
- UI tweaks such as card layout or navigation styles belong in `app/layout.py`.
- AI-powered insights are generated via `core.summary.generate_ai_summary`, with analytics inputs supplied by `core.summary_service`.

## Advanced analytics

- `analytics/categorize.py` normalises merchant descriptions so subscriptions and duplicates can be grouped reliably.
- `analytics/recurring.py` detects recurring transactions (weekly, bi-weekly, monthly) and surfaces potential duplicate charges.
- `analytics/forecast.py` estimates cash-flow runway using recent pay cycles, upcoming commitments, and projected spend.

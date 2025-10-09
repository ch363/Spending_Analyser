# PlainSpend Dashboard

A Streamlit dashboard for exploring synthetic spending insights, featuring polished card-based layouts and interactive Plotly visualisations.

## Project structure

- `app/main.py` — Streamlit entrypoint orchestrating layout and data loading.
- `analytics/` — Merchant normalisation, recurring detection, and projection helpers.
- `core/summary_service.py` — Core data aggregation pipeline for dashboard metrics.
- `core/summary.py` — High-level summary APIs including AI insights helpers.
- `core/data_loader.py` — Cached CSV reader powering the analytics stack.
- `core/models.py` — Typed data structures shared across the dashboard.
- `lib/` — Synthetic data generation utilities for seeding the app.
- `ui/components.py` — Reusable UI building blocks (cards, global styles).
- `ui/charts.py` — Centralised Plotly chart builders for consistency across the app.
- `data/seed.csv` — Seed dataset used to power the dashboard.
- `tests/` — Pytest suite scaffold with smoke checks for the app package.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your OpenAI credentials:

   - Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`.
   - Update the `api_key` value with your OpenAI key and, if needed, override the model or base URL.
   - Keep the real `secrets.toml` out of version control.

3. Launch the dashboard:

   ```bash
   streamlit run app/main.py
   ```

### Running tests

Install development dependencies (for example, `pip install pytest`) and run:

```bash
pytest
```

### AI client overrides

During tests you can inject a stand-in OpenAI client via the `client_factory` parameter on `core.generate_ai_summary`, avoiding real API calls.

## Pages

- **Overview** — The main dashboard with monthly metrics, charts, and AI highlights.
- **Insights** — An AI-guided deep dive featuring the cash-flow runway gauge, recurring subscription watchlist, duplicate charge alerts, and top category movers. Access it from the navigation bar at the top of the app.

## Development tips

- Chart styling lives in `ui/charts.py`; update there to keep visuals consistent.
- Cached data loading follows Streamlit best practices via `@st.cache_data` to avoid redundant CSV reads.
- UI tweaks such as card layout or navigation styles belong in `ui/components.py`.
- AI-powered insights are generated via `lib/ai_summary.py`, with a runtime fallback to heuristic insights if the API is unavailable.

## Advanced analytics

- `lib/categorize.py` normalises merchant descriptions so that subscriptions and duplicates can be grouped reliably.
- `lib/recurring.py` detects recurring transactions (weekly, bi-weekly, monthly) and surfaces potential duplicate charges.
- `lib/forecast.py` estimates cash-flow runway using recent pay cycles, upcoming commitments, and projected spend.

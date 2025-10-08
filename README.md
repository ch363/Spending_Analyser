# Spending Analyser

Scaffold for a Streamlit-based personal finance dashboard. Build features that ingest transaction data, classify spending, detect recurring payments, surface anomalies, and summarize monthly trends.

## Getting Started

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies via `pip install -r requirements.txt` once populated.
3. Install Streamlit extras (optional) such as `pip install streamlit-extras` if you plan to extend UI widgets.
4. Launch the Streamlit app with `streamlit run app.py`. The homepage is pre-styled and ready to demo without any backend wiring.

## Project Structure

- `app.py` — Streamlit entry point for the user interface.
- `data/seed.csv` — Optional seed dataset for bootstrapping development and demos.
- `lib/` — Library modules for synthetic data, categorization, recurring detection, forecasting, and summarization.
- `prompts/monthly_summary.txt` — Prompt template for LLM-powered summary generation.
- `requirements.txt` — Python dependency manifest.

## Frontend preview

The default landing page takes visual cues from Trading 212: a clean white card with rounded corners, deep blue highlights, and a light grey canvas background. It ships with static summary metrics, a trend chart, and a mini transaction table so you can showcase the look and feel before wiring real data.

- Upload and rules buttons are decorative only—hook them up once your ingestion pipeline is ready.
- Trend and table sections rely on placeholder data frames; replace them with live feeds when your backend is connected.
- Global theming is handled via CSS injected in `app.py`. Adjust palette constants there to experiment with new colourways.

## Next Steps

- Replace placeholder comments with real implementations in each module.
- Populate `requirements.txt` with the libraries you adopt.
- Document architectural decisions and workflows as they evolve.

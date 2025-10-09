# ADR 0001: UI and Analytics Reorganisation

## Status

Accepted (2025-10-09)

## Context

The PlainSpend dashboard had organically grown around a single `lib/` package that mixed analytics
helpers, Streamlit layout snippets, and AI-facing utilities. As the app evolved we introduced
additional pages, richer analytics, and AI summaries, which made the monolithic package difficult to
navigate, test, and extend. Configuration was handled imperatively through Streamlit secrets with no
centralised fallback or validation, and prompt templates were accessed through hard-coded file paths.

## Decision

We introduced a dedicated module layout:

- `analytics/` now contains stateless data-science helpers (categorisation, recurring detection,
  forecasting, and summary preparation).
- `core/` houses the orchestration layer (`summary_service`, `summary`, typed models, and data
  loaders) that converts raw transactions into dashboard-ready structures and AI payloads.
- `visualization/` centralises Plotly chart builders, enabling a single source of styling truth for
  Streamlit pages.
- `config/settings.py` uses `pydantic-settings` to supply validated OpenAI credentials that fall back
  to environment variables when Streamlit secrets are unavailable.
- Prompt files are stored beneath `prompts/` to keep AI templates versioned alongside code.

## Consequences

- Feature work can target the relevant package without cross-cutting edits in a monolithic `lib/`.
- Analytics helpers are easier to unit test in isolation, enabling Phase 6 test coverage.
- The Streamlit app gains a single configuration entrypoint with typed accessors instead of ad-hoc
  environment checks.
- Teams onboarding to the project can rely on the README structure section and ADR history to
  understand why modules live where they do.
- Existing imports from `lib.*` must be updated to the new namespaces (`analytics.*` or `core.*`),
  but this one-time change unlocks a clearer separation of concerns.

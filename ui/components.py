from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


def inject_css() -> None:
    """Inject global card styling for the PlainSpend dashboard."""

    st.markdown(
        """
        <style>
          :root {
            --gap: 16px;
            --radius: 12px;
            --card-bg: #FFFFFF;
            --border: #E6EAF2;
            --shadow: 0 1px 2px rgba(16, 24, 40, 0.05), 0 1px 3px rgba(16, 24, 40, 0.06);
          }

          body, [data-testid="stAppViewContainer"] > .main {
            background: #F4F6FB;
          }

          .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
          }

          .ps-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 2rem;
            padding: 0.9rem 0;
          }

          .ps-nav__brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.5rem;
            font-weight: 700;
            color: #0B3FD6;
          }

          .ps-nav__links {
            display: flex;
            align-items: center;
            gap: 1.8rem;
          }

          .ps-nav__link,
          .ps-nav__link:visited {
            position: relative;
            display: inline-flex;
            align-items: center;
            font-weight: 600;
            color: #5C6478;
            text-decoration: none;
            transition: color 0.2s ease;
          }

          .ps-nav__link:hover {
            color: #1D4ED8;
          }

          .ps-nav__link.is-active {
            color: #1D4ED8;
          }

          .ps-nav__link.is-active::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: -8px;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, #1D4ED8, #0EA5E9);
          }

          .ps-nav__link.is-disabled {
            color: #B7C1D9;
            cursor: not-allowed;
            pointer-events: none;
            opacity: 0.7;
          }

          @media (max-width: 768px) {
            .ps-nav {
              flex-direction: column;
              align-items: flex-start;
              gap: 0.75rem;
            }

            .ps-nav__links {
              gap: 1.2rem;
              flex-wrap: wrap;
            }
          }

          .ps-card-anchor {
            display: none;
          }

          [data-testid="stVerticalBlock"]:has(> .ps-card-anchor) {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 16px;
            margin-bottom: var(--gap);
            display: flex;
            flex-direction: column;
            gap: 12px;
          }

          .ps-card__head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 4px;
            flex-wrap: wrap;
          }

          .ps-card__title {
            font-size: 1.05rem;
          }

          .ps-chip {
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid #D6DEFF;
            background: #F0F4FF;
            color: #3346FF;
            white-space: nowrap;
          }

          .ps-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
            gap: var(--gap);
          }

          .ps-progress-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 600;
            color: #1F2937;
            margin-top: 12px;
            margin-bottom: 4px;
          }

          .ps-insights {
            margin: 0;
            padding-left: 1.1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            color: #4B5563;
          }

          .ps-insights li strong {
            color: #111827;
          }

          @media (min-width: 1200px) {
            [data-testid="stVerticalBlock"]:has(> .ps-card-anchor) {
              padding: 24px;
            }
            :root {
              --gap: 24px;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str, suffix: str | None = None):
    """Render content inside a reusable PlainSpend card."""

    chip_html = f'<span class="ps-chip">{suffix}</span>' if suffix else ""
    container = st.container()
    with container:
        st.markdown('<div class="ps-card-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ps-card__head"><span class="ps-card__title">{title}</span>'
            f'{chip_html}</div>',
            unsafe_allow_html=True,
        )
        yield

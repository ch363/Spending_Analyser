"""Shared layout primitives for the PlainSpend Streamlit app."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as components_html


@dataclass(frozen=True)
class NavigationLink:
    slug: str
    label: str
    enabled: bool = True


NAV_LINKS: tuple[NavigationLink, ...] = (
    NavigationLink("overview", "Dashboard", True),
    NavigationLink("insights", "Insights", True),
    NavigationLink("help", "Help", False),
    NavigationLink("settings", "Settings", False),
)


def inject_css() -> None:
    """Inject global CSS tokens and component styling into the Streamlit app."""

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

          .ps-sidebar-links {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-top: 0.75rem;
          }

          .ps-sidebar-link,
          .ps-sidebar-link:visited {
            color: #2563eb;
            text-decoration: none;
            font-weight: 600;
          }

          .ps-sidebar-link.is-active {
            color: #1d4ed8;
          }

          .ps-sidebar-link:hover {
            text-decoration: underline;
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


def sidebar_link(label: str, page: str, active_page: str, month_key: str | None) -> str:
    """Return sidebar link markup with active state handling."""

    is_active = page == active_page
    css_classes = "ps-sidebar-link" + (" is-active" if is_active else "")
    aria_current = ' aria-current="page"' if is_active else ""
    href = f"?page={page}"
    if month_key:
        href += f"&month={month_key}"
    return f"<a href='{href}' class='{css_classes}'{aria_current} target='_self'>{label}</a>"


def render_navbar(active_page: str, month_key: str | None) -> None:
    """Render the dashboard navigation bar with active state."""

    link_markup: list[str] = []
    for link in NAV_LINKS:
        css_class = "ps-nav__link"
        aria_current = ""
        if link.slug == active_page:
            css_class += " is-active"
            aria_current = ' aria-current="page"'

        if link.enabled:
            href = f"?page={link.slug}"
            if month_key:
                href += f"&month={month_key}"
            link_markup.append(
                f'<a class="{css_class}" href="{href}"{aria_current} data-page="{link.slug}" target="_self">{link.label}</a>'
            )
        else:
            link_markup.append(f'<span class="{css_class} is-disabled">{link.label}</span>')

    st.markdown(
        f"""
        <nav class="ps-nav">
            <div class="ps-nav__brand">PlainSpend</div>
            <div class="ps-nav__links">{''.join(link_markup)}</div>
        </nav>
        """,
        unsafe_allow_html=True,
    )
    _enforce_same_tab_navigation()


def render_sidebar_filters(
    month_options: list[str],
    active_page: str,
    selected_month: str | None,
) -> str | None:
    """Render the sidebar filters and return the chosen month."""

    if not month_options:
        st.sidebar.info("No transactions available yet.")
        st.session_state.pop("month_selector", None)
        return None

    candidate = selected_month if selected_month in month_options else None
    if candidate is None:
        stored = st.session_state.get("month_selector")
        if stored in month_options:
            candidate = stored
    if candidate is None:
        candidate = month_options[-1]

    try:
        default_index = month_options.index(candidate)
    except ValueError:
        default_index = len(month_options) - 1

    with st.sidebar:
        st.markdown("### Filters")
        chosen_month = st.selectbox(
            "Month",
            month_options,
            index=max(default_index, 0),
            key="month_selector",
            format_func=lambda key: pd.Period(key, freq="M").strftime("%B %Y"),
        )

        active_month = chosen_month

        st.markdown("---")
        st.markdown("### More")
        st.markdown(
            "<div class='ps-sidebar-links'>"
            f"{sidebar_link('Help', 'help', active_page, active_month)}"
            f"{sidebar_link('Settings', 'settings', active_page, active_month)}"
            "</div>",
            unsafe_allow_html=True,
        )

    _enforce_same_tab_navigation()
    return chosen_month


def _enforce_same_tab_navigation() -> None:
    """Ensure navigation links stay within the same browser tab."""

    components_html(
        """
        <script>
        (function() {
          if (window.parent && !window.parent.__psNavSameTab) {
            window.parent.__psNavSameTab = true;
            const enforce = () => {
              const anchors = window.parent.document.querySelectorAll('a.ps-nav__link, a.ps-sidebar-link');
              anchors.forEach((anchor) => {
                if (anchor.target && anchor.target.toLowerCase() !== '_self') {
                  anchor.target = '_self';
                }
              });
            };
            enforce();
            const observer = new MutationObserver(enforce);
            observer.observe(window.parent.document.body, { childList: true, subtree: true });
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def determine_active_page(valid_pages: Iterable[str]) -> str:
    """Determine the active page from the query params or session state."""

    params = st.query_params
    default_page = st.session_state.get("active_page", "overview")
    raw_page = params.get("page", default_page)
    if isinstance(raw_page, list):
        raw_page = raw_page[0]

    page = raw_page if raw_page in set(valid_pages) else "overview"

    if st.session_state.get("active_page") != page:
        st.session_state["active_page"] = page

    current_param = params.get("page")
    if isinstance(current_param, list):
        current_param = current_param[0]

    if current_param != page:
        st.query_params["page"] = page

    return page


def sync_month_query_param(month_key: str | None) -> None:
    """Ensure the ``month`` query param mirrors the current selection."""

    current_param = st.query_params.get("month")
    if isinstance(current_param, list):
        current_param = current_param[0] if current_param else None

    if month_key:
        if current_param != month_key:
            st.query_params["month"] = month_key
    else:
        if "month" in st.query_params:
            st.query_params.pop("month")


__all__ = [
    "NavigationLink",
    "NAV_LINKS",
    "card",
    "determine_active_page",
    "inject_css",
    "render_navbar",
    "render_sidebar_filters",
    "sidebar_link",
    "sync_month_query_param",
]

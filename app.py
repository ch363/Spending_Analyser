"""Streamlit homepage scaffold with Trading 212 inspired styling."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
	page_title="PlainSpend | Overview",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="collapsed",
)


def _inject_styles() -> None:
	primary_blue = "#0C6FFD"
	deep_blue = "#063E9B"
	card_shadow = "0 32px 96px rgba(6, 62, 155, 0.14)"
	card_radius = "32px"

	st.markdown(
		f"""
		<style>
			:root {{
				--primary-blue: {primary_blue};
				--deep-blue: {deep_blue};
				--soft-gray: #f3f7fc;
			}}

			body {{
				background: var(--soft-gray);
			}}

			[data-testid='stAppViewContainer'] > .main {{
				background: var(--soft-gray);
			}}

			[data-testid='stToolbar'],
			[data-testid='stDecoration'] {{
				display: none;
			}}

			[data-testid='stHeader'] {{
				background: transparent;
			}}

			.block-container {{
				padding-top: 4.5rem;
				padding-bottom: 4.5rem;
				max-width: 1160px;
				position: relative;
			}}

			.block-container::before {{
				content: "";
				position: absolute;
				top: 1.25rem;
				bottom: 1.5rem;
				left: 50%;
				transform: translateX(-50%);
				width: min(1120px, calc(100% - 38px));
				background: linear-gradient(150deg, rgba(255, 255, 255, 0.98), rgba(240, 245, 255, 0.92));
				border-radius: {card_radius};
				box-shadow: {card_shadow};
				z-index: 0;
			}}

			.block-container > * {{
				position: relative;
				z-index: 1;
			}}

			.app-heading {{
				font-size: 2.3rem;
				font-weight: 700;
				color: var(--deep-blue);
				margin-bottom: 0.25rem;
			}}

			.app-subheading {{
				font-size: 1.05rem;
				color: #6d7a8f;
				margin-bottom: 1.5rem;
			}}

			.tag-pill {{
				display: inline-flex;
				align-items: center;
				gap: 0.35rem;
				background: rgba(12, 111, 253, 0.1);
				color: var(--primary-blue);
				border-radius: 999px;
				padding: 0.35rem 0.85rem;
				font-size: 0.85rem;
				font-weight: 600;
			}}

		</style>
		""",
		unsafe_allow_html=True,
	)


def _render_header() -> None:
	st.markdown("<div class='app-heading'>PlainSpend Dashboard</div>", unsafe_allow_html=True)
	st.markdown(
		"<div class='app-subheading'>Track your savings, stay on top of recurring bills, and spot anomalies before they hit your balance.</div>",
		unsafe_allow_html=True,
	)


def main() -> None:
	_inject_styles()
	_render_header()


if __name__ == "__main__":
	main()

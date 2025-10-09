"""Backwards-compatible wrapper for running PlainSpend via ``streamlit run app.py``."""

from app.main import main

__all__ = ["main"]


if __name__ == "__main__":
    main()

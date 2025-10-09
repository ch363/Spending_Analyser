"""Centralised configuration handling for PlainSpend."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _streamlit_section(name: str) -> Mapping[str, Any] | None:
    """Return a mapping from Streamlit secrets for the given section."""

    try:
        if hasattr(st, "secrets") and name in st.secrets:
            section = st.secrets[name]
            if isinstance(section, Mapping):
                return section
            return dict(section)
    except Exception:  # pragma: no cover - accessing secrets may fail in tests
        return None
    return None


class Settings(BaseSettings):
    """Application settings sourced from env vars and Streamlit secrets."""

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL

    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")

    @property
    def openai_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.openai_api_key:
            kwargs["api_key"] = self.openai_api_key
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        return kwargs


@lru_cache
def get_settings() -> Settings:
    """Load and cache application settings."""

    overrides: dict[str, Any] = {}
    secrets_section = _streamlit_section("openai")
    if secrets_section:
        overrides = {
            "openai_api_key": secrets_section.get("api_key")
            or secrets_section.get("OPENAI_API_KEY"),
            "openai_base_url": secrets_section.get("api_base"),
            "openai_model": secrets_section.get("model", DEFAULT_OPENAI_MODEL),
        }

    return Settings(**{k: v for k, v in overrides.items() if v is not None})

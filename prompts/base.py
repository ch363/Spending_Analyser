"""Prompt loading utilities for PlainSpend AI features."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["PromptTemplate", "load_prompt", "get_prompt_text"]


@dataclass(frozen=True)
class PromptTemplate:
    """Represents a prompt template with the resolved text content."""

    name: str
    content: str


PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt(name: str) -> PromptTemplate:
    """Load a prompt template by stem name (without extension)."""

    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    return PromptTemplate(name=name, content=content)


def get_prompt_text(name: str) -> str:
    """Return raw prompt text for convenience."""

    return load_prompt(name).content

"""Backwards-compatible re-export for legacy layout helpers.

Layout primitives now live under ``app.layout``. This module keeps the old
``ui.components`` import path functioning during migration.
"""

from app.layout import *  # noqa: F401,F403

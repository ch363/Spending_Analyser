"""Backwards-compatible re-export for legacy imports.

The chart implementations now live under ``visualization.charts``. This module
simply re-exports the public API so older imports continue to function while we
migrate callers.
"""

from visualization.charts import *  # noqa: F401,F403


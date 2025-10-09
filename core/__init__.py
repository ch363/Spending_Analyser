"""Core domain package for the PlainSpend application."""

from .models import DashboardData, DailyForecastPoint, MonthlySummary, ProgressRow, ProjectionResult, VendorRow
from .summary import AISummaryError, AISummaryRequest, build_ai_summary_request, generate_ai_summary, prepare_dashboard_data

__all__ = [
    "DashboardData",
    "DailyForecastPoint",
    "MonthlySummary",
    "ProgressRow",
    "ProjectionResult",
    "VendorRow",
    "AISummaryError",
    "AISummaryRequest",
    "build_ai_summary_request",
    "generate_ai_summary",
    "prepare_dashboard_data",
]

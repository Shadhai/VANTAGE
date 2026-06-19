"""API package for lightweight graph and dashboard endpoints."""

from .graph_api import app as graph_app
from .dashboard_api import app as dashboard_app

__all__ = ["graph_app", "dashboard_app"]

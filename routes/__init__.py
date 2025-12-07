"""
Routes package for GrokRere application.
"""

from routes.api_routes import router as api_router
from routes.main_routes import router as main_router

__all__ = ["api_router", "main_router"]

# app/api/__init__.py
"""
API routers for the Document Intelligence System.
"""
from fastapi import APIRouter

from app.api.ingest import router as ingest_router
from app.api.layout import router as layout_router

# Create main router
api_router = APIRouter()

# Include all routers
api_router.include_router(ingest_router)
api_router.include_router(layout_router)

__all__ = ["api_router"]
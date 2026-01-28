"""
Main FastAPI application - WITH COMPATIBILITY FIXES
"""

# 1. Apply NumPy compatibility fix FIRST
try:
    from app.compat.numpy_fix import *
except ImportError:
    # Inline fix if module not found
    import numpy as np
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str, np.void]
        }

import os
import sys
from pathlib import Path

# ============================================================================
# POLLER PATH FIX FOR WINDOWS
# ============================================================================
# Add poppler to PATH at startup to ensure pdf2image can find it
poppler_paths_to_check = [
    r"C:\Users\surya\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin",
    r"C:\poppler\bin",
    r"C:\Program Files\poppler\bin",
]

for poppler_path in poppler_paths_to_check:
    if Path(poppler_path).exists():
        if poppler_path not in os.environ['PATH']:
            os.environ['PATH'] = poppler_path + ';' + os.environ['PATH']
            print(f"✅ Added poppler to PATH: {poppler_path}")
        else:
            print(f"✅ Poppler already in PATH: {poppler_path}")
        break
else:
    print("⚠ Warning: Poppler not found in common locations. Please set POPPLER_PATH in .env file")

# ============================================================================
# SET ENVIRONMENT VARIABLES TO PREVENT TRAINING
# ============================================================================
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['ULTRA_VERBOSE'] = 'False'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config.settings import settings

# Import Phase 1 router (safe)
from app.api.ingest import router as ingest_router

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.API_VERSION,
    description="Multi-Modal Document Intelligence System - All Phases"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Phase 1 router (always safe)
app.include_router(ingest_router)

# Health check endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.API_VERSION,
        "phases": [
            "Phase 1: Document Ingestion",
            "Phase 2: Layout Analysis", 
            "Phase 3: OCR",
            "Phase 4: Multi-Modal Agents"
        ]
    }

# ============================================================================
# LAZY LOAD PHASE 2, 3 & 4 ROUTERS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize Phase 2, 3 and 4 routers on startup."""
    # Phase 2: Layout Analysis
    try:
        from app.api.layout import router as layout_router
        app.include_router(layout_router, prefix="/layout", tags=["layout"])
        print("✅ Phase 2 (Layout Analysis) router loaded")
    except Exception as e:
        print(f"⚠️  Phase 2 router failed: {e}")
        @app.get("/layout/fallback")
        async def layout_fallback():
            return {"error": "Layout analysis temporarily unavailable"}
    
    # Phase 3: OCR
    try:
        from app.api.ocr import router as ocr_router
        app.include_router(ocr_router)  # No prefix needed
        print("✅ Phase 3 (OCR) router loaded")
    except Exception as e:
        print(f"⚠️  Phase 3 router failed: {e}")
        @app.get("/ocr/fallback")
        async def ocr_fallback():
            return {"error": "OCR temporarily unavailable"}
    
    # Phase 4: Multi-Modal Agents
    try:
        from app.api.agents import router as agents_router
        app.include_router(agents_router)  # Already has /agents prefix
        print("✅ Phase 4 (Multi-Modal Agents) router loaded")
        print("   - Vision Agent: Layout intelligence")
        print("   - Text Agent: Semantic extraction")
        print("   - Fusion Agent: Multi-modal alignment")
        print("   - Validation Agent: Confidence scoring")
        print("   - LLM: Qwen-2.5-Instruct (open-source)")
        print("   - Vector DB: Qdrant for RAG")
    except Exception as e:
        print(f"⚠️  Phase 4 router failed: {e}")
        @app.get("/agents/fallback")
        async def agents_fallback():
            return {"error": "Agent intelligence temporarily unavailable"}

# ============================================================================
# INFORMATION ENDPOINTS
# ============================================================================
@app.get("/info")
async def service_info():
    """Get service information."""
    info = {
        "service": settings.APP_NAME,
        "phases": [
            {
                "phase": 1,
                "name": "Document Ingestion & Normalization",
                "description": "PDF upload, validation, metadata generation, and page image conversion",
                "endpoints": [
                    "POST /ingest/upload",
                    "GET /ingest/status/{document_id}"
                ]
            }
        ],
        "data_directory": str(settings.BASE_DATA_DIR.absolute()),
        "current_phase": 4
    }
    
    # Dynamically add available phases
    try:
        from app.api.layout import router as layout_router
        info["phases"].append({
            "phase": 2,
            "name": "Layout & Vision Intelligence",
            "description": "Computer vision-based layout analysis using YOLOv8",
            "endpoints": [
                "POST /layout/analyze/{document_id}",
                "GET /layout/status/{document_id}",
                "GET /layout/results/{document_id}",
                "GET /layout/model/info"
            ]
        })
    except:
        pass
    
    try:
        from app.api.ocr import router as ocr_router
        info["phases"].append({
            "phase": 3,
            "name": "OCR & Region-Level Text Extraction",
            "description": "Text extraction from detected document regions using EasyOCR",
            "endpoints": [
                "POST /ocr/process/{document_id}",
                "GET /ocr/status/{document_id}",
                "GET /ocr/results/{document_id}",
                "GET /ocr/engine/info"
            ]
        })
    except:
        pass
    
    try:
        from app.api.agents import router as agents_router
        info["phases"].append({
            "phase": 4,
            "name": "Agent-Based Multi-Modal Reasoning & Fusion",
            "description": "LLM-powered agents for vision-text fusion, validation, and vector RAG",
            "endpoints": [
                "POST /agents/run/{document_id}",
                "GET /agents/status/{document_id}",
                "GET /agents/result/{document_id}",
                "POST /agents/query",
                "GET /agents/vector/info",
                "GET /agents/llm/info"
            ]
        })
    except:
        pass
    
    return info

@app.get("/endpoints")
async def list_all_endpoints():
    """List all available API endpoints."""
    endpoints = []
    for route in app.routes:
        if hasattr(route, "methods"):
            endpoints.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"endpoints": sorted(endpoints, key=lambda x: x["path"])}

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"Starting {settings.APP_NAME}")
    print(f"Version: {settings.API_VERSION}")
    print(f"Data directory: {settings.BASE_DATA_DIR.absolute()}")
    
    if hasattr(settings, 'POPPLER_PATH') and settings.POPPLER_PATH:
        print(f"Poppler path: {settings.POPPLER_PATH}")
    
    print("\nAvailable endpoints:")
    print("- Phase 1:")
    print("  POST /ingest/upload        - Upload and process PDF")
    print("  GET  /ingest/status/{id}   - Check processing status")
    
    # Check Phase 2 availability
    try:
        from app.api.layout import router as layout_router
        print("- Phase 2:")
        print("  POST /layout/analyze/{id}  - Start layout analysis")
        print("  GET  /layout/status/{id}   - Check layout status")
        print("  GET  /layout/results/{id}  - Get layout results")
        print("  GET  /layout/model/info    - Get YOLO model info")
    except:
        print("- Phase 2: Not available (import error)")
    
    # Check Phase 3 availability
    try:
        from app.api.ocr import router as ocr_router
        print("- Phase 3:")
        print("  POST /ocr/process/{id}     - Start OCR processing")
        print("  GET  /ocr/status/{id}      - Check OCR status")
        print("  GET  /ocr/results/{id}     - Get OCR results")
        print("  GET  /ocr/engine/info      - Get OCR engine info")
    except:
        print("- Phase 3: Not available (import error)")
    
    # Check Phase 4 availability
    try:
        from app.api.agents import router as agents_router
        print("- Phase 4:")
        print("  POST /agents/run/{id}      - Start multi-agent reasoning")
        print("  GET  /agents/status/{id}   - Check agent pipeline status")
        print("  GET  /agents/result/{id}   - Get structured understanding")
        print("  POST /agents/query         - Multi-modal RAG query")
        print("  GET  /agents/vector/info   - Vector DB information")
        print("  GET  /agents/llm/info      - LLM configuration")
    except:
        print("- Phase 4: Not available (import error)")
    
    print("- System:")
    print("  GET  /health               - Health check")
    print("  GET  /info                 - Service info")
    print("  GET  /endpoints            - List all endpoints")
    print("  GET  /docs                 - API documentation")
    print("=" * 70)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
"""
Main FastAPI application - Complete with routing fix
"""
import os
import sys
from pathlib import Path

# ============================================================================
# POLLER PATH FIX FOR WINDOWS
# ============================================================================
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
# SET ENVIRONMENT VARIABLES
# ============================================================================
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['ULTRA_VERBOSE'] = 'False'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Phase 1 router (always safe)
app.include_router(ingest_router)

# Debug endpoint to check what routes are loaded
@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to see all registered routes."""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"routes": sorted(routes, key=lambda x: x["path"])}

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
            "Phase 4: Multi-Modal Agents",
            "Phase 5: RAG & Query"
        ]
    }

# ============================================================================
# LAZY LOAD ALL ROUTERS - FIXED VERSION
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize all routers on startup."""
    print("\n" + "="*60)
    print("DEBUG: Loading all routers...")
    print("="*60)
    
    # Phase 2: Layout Analysis - FIXED: Router already has /layout prefix
    try:
        from app.api.layout import router as layout_router
        app.include_router(layout_router)  # FIXED: No duplicate prefix
        print("✅ Phase 2 (Layout Analysis) router loaded")
        print(f"   Routes: {len(layout_router.routes)}")
        
        # Debug: Print layout routes
        for route in layout_router.routes:
            if hasattr(route, "path"):
                print(f"   - {route.path}")
    except Exception as e:
        print(f"⚠️  Phase 2 router failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 3: OCR
    try:
        from app.api.ocr import router as ocr_router
        app.include_router(ocr_router)  # Already has /ocr prefix
        print("✅ Phase 3 (OCR) router loaded")
        print(f"   Routes: {len(ocr_router.routes)}")
    except Exception as e:
        print(f"⚠️  Phase 3 router failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 4: Multi-Modal Agents
    try:
        from app.api.agents import router as agents_router
        app.include_router(agents_router)  # Already has /agents prefix
        print("✅ Phase 4 (Multi-Modal Agents) router loaded")
        print(f"   Routes: {len(agents_router.routes)}")
    except Exception as e:
        print(f"⚠️  Phase 4 router failed: {e}")
    
    # Phase 5: RAG & Query
    try:
        from app.api.rag_query import router as rag_router
        app.include_router(rag_router)  # Already has /rag prefix
        print("✅ Phase 5 (RAG & Query) router loaded")
        print(f"   Routes: {len(rag_router.routes)}")
    except Exception as e:
        print(f"⚠️  Phase 5 (RAG) router failed: {e}")
    
    print("="*60)
    print("All routers loaded")
    print("="*60 + "\n")
    
    # Print all registered routes for debugging
    print("DEBUG: All registered routes in app:")
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"  {route.path} - {list(route.methods)}")

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
        "current_phase": 5
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
    
    try:
        from app.api.rag_query import router as rag_router
        info["phases"].append({
            "phase": 5,
            "name": "RAG & Multi-Modal Query",
            "description": "Retrieval-Augmented Generation with multi-modal document understanding",
            "endpoints": [
                "POST /rag/query",
                "POST /rag/query/simple",
                "POST /rag/index/{document_id}",
                "GET /rag/index/status/{document_id}",
                "GET /rag/stats"
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
    
    print("\nAvailable endpoints:")
    print("- Phase 1 (Document Ingestion):")
    print("  POST /ingest/upload        - Upload and process PDF")
    print("  GET  /ingest/status/{id}   - Check processing status")
    
    print("- Phase 2 (Layout Analysis):")
    print("  POST /layout/analyze/{id}  - Start layout analysis")
    print("  GET  /layout/status/{id}   - Check layout status")
    print("  GET  /layout/results/{id}  - Get layout results")
    
    print("- Phase 3 (OCR):")
    print("  POST /ocr/process/{id}     - Start OCR processing")
    print("  GET  /ocr/status/{id}      - Check OCR status")
    print("  GET  /ocr/results/{id}     - Get OCR results")
    
    print("- Phase 4 (Multi-Modal Agents):")
    print("  POST /agents/run/{id}      - Start multi-agent reasoning")
    print("  GET  /agents/status/{id}   - Check agent pipeline status")
    print("  GET  /agents/result/{id}   - Get structured understanding")
    print("  POST /agents/query         - Multi-modal RAG query")
    
    print("- Phase 5 (RAG & Query):")
    print("  POST /rag/query            - Multi-modal RAG query")
    print("  POST /rag/index/{id}       - Index document for RAG")
    print("  GET  /rag/stats            - Get RAG system stats")
    
    print("\nSystem endpoints:")
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
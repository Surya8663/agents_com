"""
FastAPI endpoints for Phase 4: Agent-Based Multi-Modal Reasoning.
"""
import uuid
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.config.settings import settings
from app.agents.graph import agent_orchestrator
from app.vectorstore.qdrant_client import qdrant_store
from app.llm.embeddings import embedding_generator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])  # This line was missing!


# Request/Response models
class AgentRunRequest(BaseModel):
    """Request model for running agents."""
    force_rerun: bool = False


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    document_id: Optional[str] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    results: list
    document_context: Optional[str] = None


# In-memory tracking (in production, use Redis or database)
agent_tasks = {}


async def _run_agent_pipeline(document_id: uuid.UUID):
    """Background task to run agent pipeline."""
    try:
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        
        # Load Phase 2 layout data
        layout_summary = doc_dir / "layout" / "document_layout_summary.json"
        if not layout_summary.exists():
            raise ValueError(f"Layout data not found for {document_id}")
        
        with open(layout_summary, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
        
        # Load Phase 3 OCR data
        ocr_dir = doc_dir / "ocr"
        ocr_files = list(ocr_dir.glob("page_*_ocr.json"))
        
        if not ocr_files:
            # Wait for OCR to complete
            logger.info(f"⏳ Waiting for OCR results for {document_id}...")
            for attempt in range(30):  # 30 * 2 = 60 seconds max
                ocr_files = list(ocr_dir.glob("page_*_ocr.json"))
                if ocr_files:
                    break
                time.sleep(2)
            
            if not ocr_files:
                raise ValueError(f"No OCR results found for {document_id} after waiting")
        
        # Compile OCR data
        ocr_pages = []
        for ocr_file in sorted(ocr_files, key=lambda x: int(x.stem.split('_')[1])):
            with open(ocr_file, "r", encoding="utf-8") as f:
                ocr_pages.append(json.load(f))
        
        ocr_data = {
            "document_id": str(document_id),
            "pages": ocr_pages
        }
        
        # Run agent pipeline
        result = await agent_orchestrator.run(str(document_id), layout_data, ocr_data)
        
        # Save agent results
        agent_dir = doc_dir / "agents"
        agent_dir.mkdir(exist_ok=True)
        
        result_file = agent_dir / "agent_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        
        # Update task status
        agent_tasks[str(document_id)] = {
            "status": "completed",
            "result": result
        }
        
        logger.info(f"Agent pipeline completed for {document_id}")
        
    except Exception as e:
        logger.error(f"Agent pipeline failed for {document_id}: {e}")
        agent_tasks[str(document_id)] = {
            "status": "failed",
            "error": str(e)
        }


@router.post("/run/{document_id}")
async def run_agents(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    request: Optional[AgentRunRequest] = None
):
    """
    Start agent-based multi-modal reasoning for a document.
    """
    # Check document exists
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    if not doc_dir.exists():
        raise HTTPException(404, detail=f"Document {document_id} not found")
    
    # Check layout exists
    layout_dir = doc_dir / "layout"
    if not layout_dir.exists():
        raise HTTPException(400, detail="Run layout analysis first (/layout/analyze)")
    
    # Check OCR exists or is processing
    ocr_dir = doc_dir / "ocr"
    if not ocr_dir.exists():
        raise HTTPException(400, detail="Run OCR processing first (/ocr/process)")
    
    # Check if already running or completed
    doc_id_str = str(document_id)
    if doc_id_str in agent_tasks:
        task_status = agent_tasks[doc_id_str]["status"]
        
        if task_status == "processing":
            return {
                "message": f"Agent pipeline already running for document {document_id}",
                "document_id": document_id,
                "status": "processing"
            }
        elif task_status == "completed" and not (request and request.force_rerun):
            return {
                "message": f"Agent pipeline already completed for document {document_id}",
                "document_id": document_id,
                "status": "completed",
                "has_results": True
            }
    
    # Start background task
    agent_tasks[doc_id_str] = {"status": "processing"}
    background_tasks.add_task(_run_agent_pipeline, document_id)
    
    return {
        "message": "Agent pipeline started",
        "document_id": document_id,
        "status": "processing",
        "agents": ["VisionAgent", "TextAgent", "FusionAgent", "ValidationAgent"],
        "workflow": "LangGraph multi-agent orchestration",
        "llm": "Qwen-2.5-Instruct (open-source)",
        "vector_db": "Qdrant"
    }


@router.get("/status/{document_id}")
async def get_agent_status(document_id: uuid.UUID):
    """Get agent pipeline execution status."""
    doc_id_str = str(document_id)
    
    if doc_id_str not in agent_tasks:
        # Check if results exist on disk
        agent_file = settings.DOCUMENTS_DIR / str(document_id) / "agents" / "agent_results.json"
        if agent_file.exists():
            return {
                "document_id": document_id,
                "status": "completed",
                "has_results": True,
                "stored_on_disk": True
            }
        
        return {
            "document_id": document_id,
            "status": "not_started",
            "has_results": False
        }
    
    task_info = agent_tasks[doc_id_str]
    
    response = {
        "document_id": document_id,
        "status": task_info["status"]
    }
    
    if task_info["status"] == "completed":
        response["has_results"] = True
        if "result" in task_info:
            result = task_info["result"]
            response["confidence_score"] = result.get("final_output", {}).get("confidence_score", 0.0)
            response["agents_executed"] = result.get("agents_executed", [])
    
    elif task_info["status"] == "failed":
        response["error"] = task_info.get("error", "Unknown error")
    
    return response


@router.get("/result/{document_id}")
async def get_agent_result(document_id: uuid.UUID):
    """Get agent pipeline results."""
    # Try memory first
    doc_id_str = str(document_id)
    if doc_id_str in agent_tasks and agent_tasks[doc_id_str]["status"] == "completed":
        if "result" in agent_tasks[doc_id_str]:
            return agent_tasks[doc_id_str]["result"]
    
    # Try disk
    agent_file = settings.DOCUMENTS_DIR / str(document_id) / "agents" / "agent_results.json"
    if agent_file.exists():
        with open(agent_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    raise HTTPException(404, detail=f"No agent results found for document {document_id}")


@router.get("/prerequisites/{document_id}")
async def check_prerequisites(document_id: uuid.UUID):
    """Check if all prerequisites for agent pipeline are met."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(404, detail=f"Document {document_id} not found")
    
    # Check layout
    layout_dir = doc_dir / "layout"
    layout_exists = layout_dir.exists()
    layout_summary_exists = (layout_dir / "document_layout_summary.json").exists() if layout_exists else False
    
    # Check OCR
    ocr_dir = doc_dir / "ocr"
    ocr_exists = ocr_dir.exists()
    ocr_files_exist = len(list(ocr_dir.glob("page_*_ocr.json"))) > 0 if ocr_exists else False
    
    return {
        "document_id": str(document_id),
        "prerequisites": {
            "document_exists": True,
            "layout_analysis_done": layout_summary_exists,
            "ocr_processing_done": ocr_files_exist,
            "all_prerequisites_met": layout_summary_exists and ocr_files_exist
        },
        "next_steps": {
            "layout": "Run /layout/analyze" if not layout_summary_exists else "✓ Complete",
            "ocr": "Run /ocr/process" if not ocr_files_exist else "✓ Complete",
            "agents": "Run /agents/run" if layout_summary_exists and ocr_files_exist else "Wait for prerequisites"
        }
    }


@router.post("/query")
async def rag_query(request: QueryRequest):
    """
    Multi-modal RAG query over document intelligence.
    
    Supports queries like:
    - "What does the table under Figure 2 represent?"
    - "Extract invoice total and validate visually"
    - "What are the key findings in the report?"
    """
    # Generate query embedding
    query_embedding = embedding_generator.generate_text_embedding(request.query)
    
    if not query_embedding or all(v == 0 for v in query_embedding):
        raise HTTPException(500, detail="Failed to generate query embedding")
    
    # Search in Qdrant
    results = qdrant_store.search_similar(
        query_embedding=query_embedding,
        document_id=request.document_id,
        limit=request.top_k
    )
    
    # If we have a specific document, try to get context
    document_context = None
    if request.document_id:
        doc_dir = settings.DOCUMENTS_DIR / request.document_id
        agent_file = doc_dir / "agents" / "agent_results.json"
        
        if agent_file.exists():
            with open(agent_file, "r", encoding="utf-8") as f:
                agent_result = json.load(f)
                document_context = agent_result.get("final_output", {}).get("document_type", "unknown")
    
    return QueryResponse(
        query=request.query,
        results=results,
        document_context=document_context
    )


@router.get("/vector/info")
async def get_vector_db_info():
    """Get vector database information."""
    try:
        # Try to get Qdrant info
        if qdrant_store.client:
            collections = qdrant_store.client.get_collections()
            
            # Get collection stats
            stats = qdrant_store.client.get_collection(
                collection_name=qdrant_store.collection_name
            )
            
            return {
                "vector_db": "Qdrant",
                "status": "connected",
                "collections": [c.name for c in collections.collections],
                "current_collection": qdrant_store.collection_name,
                "points_count": stats.points_count if hasattr(stats, 'points_count') else 0,
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_size": 384
            }
        else:
            return {
                "vector_db": "Qdrant",
                "status": "not_connected",
                "error": "Qdrant client not initialized"
            }
            
    except Exception as e:
        return {
            "vector_db": "Qdrant",
            "status": "error",
            "error": str(e)
        }


@router.get("/llm/info")
async def get_llm_info():
    """Get LLM configuration information."""
    from app.llm.client import llm_client
    
    return {
        "llm_provider": "open_source",
        "model": llm_client.model,
        "base_url": llm_client.base_url,
        "status": "configured",
        "supported_agents": ["VisionAgent", "TextAgent", "FusionAgent", "ValidationAgent"],
        "note": "Using Qwen-2.5-Instruct via OpenAI-compatible API"
    }
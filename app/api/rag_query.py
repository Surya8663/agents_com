"""
Multi-modal RAG query API endpoints.
"""
import json
import logging
import re
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

from app.rag.multimodal_schema import RAGQuery, RAGResponse, Modality
from app.rag.retriever import retriever
from app.rag.indexer import indexer
from app.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


class SimpleQueryRequest(BaseModel):
    """Simple query request for UI."""
    query: str
    document_id: Optional[str] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    """Query response for UI."""
    query: str
    answer: str
    confidence: float
    sources: list
    needs_review: bool


@router.post("/query")
async def rag_query(request: RAGQuery):
    """
    Multi-modal RAG query endpoint.
    
    Supports complex queries with modality filtering, page filtering,
    and confidence thresholds.
    """
    try:
        logger.info(f"RAG query: '{request.query}' "
                   f"(modality: {request.modality}, doc: {request.document_id})")
        
        # Execute query
        response = await retriever.query(request)
        
        return response
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query/simple")
async def simple_rag_query(request: SimpleQueryRequest):
    """
    Simple RAG query endpoint for UI.
    
    Returns simplified response suitable for user interfaces.
    """
    try:
        # Convert to RAGQuery
        rag_query = RAGQuery(
            query=request.query,
            document_id=request.document_id,
            top_k=request.top_k,
            modality=Modality.MULTIMODAL
        )
        
        # Execute query
        response = await retriever.query(rag_query)
        
        # Extract sources
        sources = []
        for result in response.results[:3]:  # Top 3 sources
            chunk = result.chunk
            source_info = {
                "page": chunk.page_number,
                "confidence": result.relevance_score,
                "type": result.evidence_type,
                "content": chunk.text or chunk.visual_description or ""
            }
            sources.append(source_info)
        
        # Determine if needs review
        needs_review = response.confidence < 0.7
        
        return QueryResponse(
            query=response.query,
            answer=response.answer or "No answer generated",
            confidence=response.confidence,
            sources=sources,
            needs_review=needs_review
        )
        
    except Exception as e:
        logger.error(f"Simple RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/index/{document_id}")
async def index_document(document_id: str, background_tasks: BackgroundTasks):
    """
    Index a document for RAG retrieval.
    
    This endpoint:
    1. Loads agent results for the document
    2. Creates multi-modal chunks
    3. Generates embeddings
    4. Stores in vector database
    """
    try:
        logger.info(f"🔄 Starting RAG indexing for document {document_id}")
        
        # Check if document exists
        doc_dir = settings.DOCUMENTS_DIR / document_id
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Check if agent results exist
        agent_file = doc_dir / "agents" / "agent_results.json"
        if not agent_file.exists():
            # Try alternative locations
            agent_file_alt = doc_dir / "agent_results.json"
            if agent_file_alt.exists():
                agent_file = agent_file_alt
            else:
                raise HTTPException(status_code=400, detail="Run agent pipeline first (/agents/run)")
        
        # Read and validate agent results WITH EMERGENCY REPAIR
        agent_results = load_and_repair_agent_results(agent_file, document_id)
        
        if not agent_results:
            raise HTTPException(
                status_code=400, 
                detail="Could not load or repair agent results. JSON is too badly broken."
            )
        
        # Validate agent results structure
        if not validate_agent_results(agent_results):
            logger.warning(f"⚠️ Agent results structure invalid for {document_id}, attempting to fix")
            agent_results = fix_agent_results_structure(agent_results, document_id)
        
        # Clean the JSON before passing to background task
        cleaned_agent_results = clean_json_structure(agent_results)
        
        # Start indexing in background with validated data
        background_tasks.add_task(index_document_background, document_id, cleaned_agent_results)
        
        return {
            "message": "Document indexing started",
            "document_id": document_id,
            "status": "processing",
            "agent_results_valid": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document indexing initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


def load_and_repair_agent_results(file_path: Path, document_id: str) -> Optional[Dict[str, Any]]:
    """Load and repair agent results with EMERGENCY fixes."""
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        logger.info(f"📄 Loading agent results for {document_id}")
        
        # Apply emergency fixes for KNOWN issues
        content = apply_emergency_json_fixes(content)
        
        # Try to parse
        try:
            agent_results = json.loads(content)
            logger.info(f"✅ Successfully loaded agent results for {document_id}")
            return agent_results
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            
            # Apply aggressive fixes
            content = apply_aggressive_json_fixes(content)
            
            try:
                agent_results = json.loads(content)
                logger.info(f"✅ Successfully loaded after aggressive fixes for {document_id}")
                return agent_results
            except json.JSONDecodeError as e2:
                logger.error(f"❌ Even aggressive fixes failed: {e2}")
                
                # Last resort: extract what we can
                return extract_partial_json(content, document_id)
                
    except Exception as e:
        logger.error(f"❌ Failed to read agent results: {e}")
        return None


def apply_emergency_json_fixes(content: str) -> str:
    """Apply emergency JSON fixes for KNOWN issues."""
    if not content:
        return "{}"
    
    # Remove control characters
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    
    # Remove markdown
    content = content.replace('```json', '').replace('```', '')
    
    # EMERGENCY FIX 1: Fix "parent": p3_r0 pattern
    def fix_parent_pattern(match):
        key = match.group(1)
        value = match.group(2)
        if re.match(r'^p\d+_r\d+$', value):
            return f'"{key}": "{value}"'
        return f'"{key}": {value}'
    
    content = re.sub(r'"parent"\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)', fix_parent_pattern, content)
    
    # EMERGENCY FIX 2: Fix all unquoted string values
    def fix_unquoted_strings(match):
        key = match.group(1)
        value = match.group(2)
        
        # Check if it needs quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")) or \
           re.match(r'^-?\d+(\.\d+)?$', value) or \
           value in ['true', 'false', 'null'] or \
           value.startswith('[') or value.startswith('{'):
            return f'"{key}": {value}'
        
        # It's a string - quote it
        return f'"{key}": "{value}"'
    
    content = re.sub(r'"([^"]+)"\s*:\s*([^,\[\]{}\s]+(?:\s+[^,\[\]{}\s]+)*)(?=[,\]}])', 
                    fix_unquoted_strings, content)
    
    # Fix trailing commas
    content = re.sub(r',\s*([}\]])', r'\1', content)
    
    return content


def apply_aggressive_json_fixes(content: str) -> str:
    """Apply aggressive JSON fixes."""
    # Try to find JSON object
    start = content.find('{')
    end = content.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        content = content[start:end+1]
    
    # Remove any text after the last }
    last_brace = content.rfind('}')
    if last_brace != -1:
        content = content[:last_brace+1]
    
    return content


def extract_partial_json(content: str, document_id: str) -> Dict[str, Any]:
    """Extract partial JSON when all else fails."""
    result = {
        "document_id": document_id,
        "status": "completed_with_errors",
        "errors": ["JSON parsing failed, using fallback data"],
        "final_output": {
            "document_id": document_id,
            "extracted_fields": {},
            "confidence_score": 0.0,
            "document_type": "unknown"
        }
    }
    
    try:
        # Try to extract at least the document type
        doc_type_match = re.search(r'"document_type"\s*:\s*"([^"]+)"', content)
        if doc_type_match:
            result["final_output"]["document_type"] = doc_type_match.group(1)
        
        # Extract confidence if present
        conf_match = re.search(r'"confidence"\s*:\s*(\d+\.?\d*)', content)
        if conf_match:
            try:
                result["final_output"]["confidence_score"] = float(conf_match.group(1))
            except:
                pass
        
        logger.info(f"📊 Extracted partial data for {document_id}")
        
    except Exception as e:
        logger.warning(f"Could not extract partial JSON: {e}")
    
    return result


def validate_agent_results(agent_results: dict) -> bool:
    """Validate agent results structure."""
    if not isinstance(agent_results, dict):
        return False
    
    # Check for required fields
    required_fields = ["document_id", "status"]
    for field in required_fields:
        if field not in agent_results:
            return False
    
    # Check if we have at least some analysis
    analysis_fields = ["vision_analysis", "text_analysis", "fused_document", "final_output"]
    has_analysis = any(field in agent_results for field in analysis_fields)
    
    return has_analysis


def fix_agent_results_structure(agent_results: dict, document_id: str) -> dict:
    """Fix agent results structure."""
    if not isinstance(agent_results, dict):
        agent_results = {}
    
    # Ensure document_id
    agent_results["document_id"] = document_id
    
    # Ensure status
    if "status" not in agent_results:
        agent_results["status"] = "completed"
    
    # Ensure timestamp
    if "timestamp" not in agent_results:
        from datetime import datetime
        agent_results["timestamp"] = datetime.now().isoformat()
    
    # Ensure final_output
    if "final_output" not in agent_results:
        agent_results["final_output"] = {
            "document_id": document_id,
            "extracted_fields": {},
            "confidence_score": 0.0,
            "validation_notes": ["Agent results structure repaired"],
            "document_type": "unknown"
        }
    
    return agent_results


def clean_json_structure(data: Any) -> Any:
    """Recursively clean JSON structure to ensure it's valid."""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Clean the key
            if not isinstance(key, str):
                key = str(key)
            
            # Clean the value
            cleaned[key] = clean_json_structure(value)
        return cleaned
    
    elif isinstance(data, list):
        return [clean_json_structure(item) for item in data]
    
    elif isinstance(data, str):
        # Clean string: remove problematic characters
        data = data.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        # Escape quotes if needed
        if '"' in data and '\\"' not in data:
            data = data.replace('"', '\\"')
        return data
    
    elif isinstance(data, (int, float, bool)):
        return data
    
    elif data is None:
        return None
    
    else:
        # Convert any other type to string
        try:
            return str(data)
        except:
            return "[Unserializable object]"


async def index_document_background(document_id: str, agent_results: dict):
    """Background task to index document with validated data."""
    try:
        logger.info(f"📝 Processing indexing for {document_id}")
        
        # Save cleaned results (for debugging)
        doc_dir = settings.DOCUMENTS_DIR / document_id
        cleaned_file = doc_dir / "agents" / "cleaned_agent_results.json"
        cleaned_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cleaned_file, 'w', encoding='utf-8') as f:
                json.dump(agent_results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved cleaned agent results for {document_id}")
        except Exception as e:
            logger.warning(f"Could not save cleaned results: {e}")
        
        # Call indexer
        success = indexer.index_document(document_id, agent_results)
        
        if success:
            logger.info(f"✅ Document {document_id} indexed successfully")
        else:
            logger.error(f"❌ Document {document_id} indexing failed")
            
    except Exception as e:
        logger.error(f"❌ Background indexing failed for {document_id}: {e}")
        import traceback
        traceback.print_exc()


@router.get("/index/status/{document_id}")
async def get_index_status(document_id: str):
    """Get indexing status for a document."""
    try:
        from app.vectorstore.qdrant_client import qdrant_store
        
        # Check if cleaned results exist (means indexing was attempted)
        doc_dir = settings.DOCUMENTS_DIR / document_id
        cleaned_file = doc_dir / "agents" / "cleaned_agent_results.json"
        
        if cleaned_file.exists():
            # Indexing was attempted
            try:
                with open(cleaned_file, 'r', encoding='utf-8') as f:
                    cleaned_data = json.load(f)
                
                # Try to check vector store
                test_embedding = [0.0] * 384  # Assuming text embedding dimension
                
                try:
                    results = qdrant_store.search_similar(
                        query_embedding=test_embedding,
                        document_id=document_id,
                        limit=1
                    )
                    
                    return {
                        "document_id": document_id,
                        "indexed": len(results) > 0,
                        "chunks_found": len(results),
                        "agent_results_valid": True,
                        "status": "indexed" if len(results) > 0 else "failed"
                    }
                except Exception as store_error:
                    return {
                        "document_id": document_id,
                        "indexed": False,
                        "chunks_found": 0,
                        "agent_results_valid": True,
                        "status": "vector_store_error",
                        "error": str(store_error)
                    }
                    
            except Exception as e:
                return {
                    "document_id": document_id,
                    "indexed": False,
                    "chunks_found": 0,
                    "agent_results_valid": False,
                    "status": "agent_results_error",
                    "error": str(e)
                }
        else:
            return {
                "document_id": document_id,
                "indexed": False,
                "chunks_found": 0,
                "agent_results_valid": False,
                "status": "not_attempted",
                "message": "Run /rag/index/{document_id} first"
            }
        
    except Exception as e:
        return {
            "document_id": document_id,
            "indexed": False,
            "chunks_found": 0,
            "agent_results_valid": False,
            "error": str(e)
        }


@router.get("/stats")
async def get_rag_stats():
    """Get RAG system statistics."""
    try:
        from app.vectorstore.qdrant_client import qdrant_store
        
        stats = {
            "system": "Multi-Modal RAG",
            "version": "1.0",
            "features": [
                "Text retrieval",
                "Vision retrieval",
                "Cross-modal search",
                "Confidence scoring",
                "Human review integration"
            ],
            "embedding_models": {
                "text": "all-MiniLM-L6-v2 (384D)",
                "vision": "CLIP ViT-B/32 (512D)"
            },
            "vector_store": "Qdrant with multi-modal collections"
        }
        
        # Try to get Qdrant stats
        if hasattr(qdrant_store, 'client') and qdrant_store.client:
            try:
                collections = qdrant_store.client.get_collections()
                stats["collections"] = [c.name for c in collections.collections]
                
                # Get points count for each collection
                points_counts = {}
                for collection in collections.collections:
                    collection_info = qdrant_store.client.get_collection(collection.name)
                    if hasattr(collection_info, 'points_count'):
                        points_counts[collection.name] = collection_info.points_count
                
                stats["points_counts"] = points_counts
                
            except Exception as e:
                stats["vector_store_error"] = str(e)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics unavailable: {str(e)}")
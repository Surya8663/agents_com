"""
Multi-modal RAG query API endpoints.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

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
        # Check if document exists
        doc_dir = settings.DOCUMENTS_DIR / document_id
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Check if agent results exist
        agent_file = doc_dir / "agents" / "agent_results.json"
        if not agent_file.exists():
            raise HTTPException(status_code=400, detail="Run agent pipeline first (/agents/run)")
        
        # Start indexing in background
        background_tasks.add_task(_index_document_background, document_id)
        
        return {
            "message": "Document indexing started",
            "document_id": document_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


async def _index_document_background(document_id: str):
    """Background task to index document."""
    try:
        success = indexer.index_document(document_id)
        
        if success:
            logger.info(f"✅ Document {document_id} indexed successfully")
        else:
            logger.error(f"❌ Document {document_id} indexing failed")
            
    except Exception as e:
        logger.error(f"Background indexing failed for {document_id}: {e}")


@router.get("/index/status/{document_id}")
async def get_index_status(document_id: str):
    """Get indexing status for a document."""
    # Check if document is indexed
    # For now, we'll check if chunks exist in vector store
    # In production, you'd have a proper indexing status tracker
    
    try:
        from app.vectorstore.qdrant_client import qdrant_store
        
        # Simple check: try to retrieve something
        test_query = "document"
        test_embedding = [0.0] * 384
        
        results = qdrant_store.search_similar(
            query_embedding=test_embedding,
            document_id=document_id,
            limit=1
        )
        
        if results:
            return {
                "document_id": document_id,
                "indexed": True,
                "chunks_found": len(results)
            }
        else:
            return {
                "document_id": document_id,
                "indexed": False,
                "chunks_found": 0
            }
        
    except Exception as e:
        return {
            "document_id": document_id,
            "indexed": False,
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
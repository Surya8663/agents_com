# app/vectorstore/qdrant_client.py - FIXED VERSION
"""
Qdrant client for multi-modal RAG - WITH FALLBACK.
"""
import os
import logging
import uuid
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector database for document embeddings with fallback."""
    
    def __init__(self, collection_name: str = "document_chunks"):
        """Initialize Qdrant client with fallback."""
        self.collection_name = collection_name
        self.client = None
        self.use_fallback = False
        
        # Try to connect to Qdrant
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client with fallback on failure."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            # Get connection details
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            
            logger.info(f"Attempting to connect to Qdrant at {host}:{port}")
            self.client = QdrantClient(host=host, port=port)
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"✅ Qdrant connected. Collections: {len(collections.collections)}")
            
            # Ensure collection exists
            self._ensure_collection()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Qdrant: {e}")
            logger.info("   Using in-memory fallback storage")
            self.use_fallback = True
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize in-memory fallback storage."""
        self.fallback_storage = {}
        logger.info("✅ Initialized in-memory vector store fallback")
    
    def _ensure_collection(self):
        """Ensure collection exists (only for real Qdrant)."""
        if self.use_fallback or not self.client:
            return
        
        try:
            from qdrant_client.http import models
            
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # Matches all-MiniLM-L6-v2
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            self.use_fallback = True
    
    def store_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks."""
        if self.use_fallback or not self.client:
            # Use fallback storage
            if document_id not in self.fallback_storage:
                self.fallback_storage[document_id] = []
            
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                chunk_with_id = {
                    "id": chunk_id,
                    **chunk,
                    "document_id": document_id
                }
                self.fallback_storage[document_id].append(chunk_with_id)
            
            logger.info(f"📦 Stored {len(chunks)} chunks in memory for {document_id}")
            return True
        
        # Use real Qdrant
        try:
            from qdrant_client.http import models
            
            points = []
            for chunk in chunks:
                point_id = str(uuid.uuid4())
                
                point = models.PointStruct(
                    id=point_id,
                    vector=chunk.get("embedding", [0.0] * 384),
                    payload={
                        "document_id": document_id,
                        "text": chunk.get("text", ""),
                        **chunk.get("metadata", {})
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"📦 Stored {len(points)} chunks in Qdrant for {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks in Qdrant: {e}")
            # Fallback to memory
            self.use_fallback = True
            return self.store_document_chunks(document_id, chunks)
    
    def search_similar(self, query_embedding: List[float], document_id: Optional[str] = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.use_fallback or not self.client:
            # Search in fallback storage
            results = []
            
            for doc_id, chunks in self.fallback_storage.items():
                if document_id and doc_id != document_id:
                    continue
                
                for chunk in chunks:
                    # Simple cosine similarity for fallback
                    chunk_embedding = chunk.get("embedding", [0.0] * 384)
                    
                    # Calculate cosine similarity
                    import numpy as np
                    
                    query_np = np.array(query_embedding)
                    chunk_np = np.array(chunk_embedding)
                    
                    if np.linalg.norm(query_np) > 0 and np.linalg.norm(chunk_np) > 0:
                        similarity = np.dot(query_np, chunk_np) / (np.linalg.norm(query_np) * np.linalg.norm(chunk_np))
                    else:
                        similarity = 0.0
                    
                    results.append({
                        "score": float(similarity),
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {}),
                        "document_id": doc_id
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
        
        # Use real Qdrant
        try:
            from qdrant_client.http import models
            
            # Build filter
            filter_condition = None
            if document_id:
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_condition,
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


# Global instance
qdrant_store = QdrantStore()
# app/vectorstore/qdrant_client.py - EXTENDED FOR MULTI-MODAL - FIXED VERSION
"""
Qdrant client for multi-modal RAG - EXTENDED VERSION with SAFE INITIALIZATION.
"""
import os
import logging
import uuid
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector database for document embeddings - EXTENDED for multi-modal."""
    
    def __init__(self, collection_name: str = "document_chunks"):
        """Initialize Qdrant client with multi-modal support - SAFE VERSION."""
        self.collection_name = collection_name
        self.client = None
        self.use_fallback = True  # Start with fallback by default
        self.multi_modal_collections = {
            "text": f"{collection_name}_text",
            "vision": f"{collection_name}_vision",
            "multimodal": f"{collection_name}_multimodal"
        }
        
        # Initialize fallback storage
        self.fallback_storage = {}
        
        # Try to connect to Qdrant, but don't crash if it fails
        self._initialize_client_safe()
    
    def _initialize_client_safe(self):
        """Initialize Qdrant client safely - don't crash on failure."""
        try:
            # Try to import Qdrant
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            # Get connection details
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            
            logger.info(f"🔄 Attempting to connect to Qdrant at {host}:{port}")
            self.client = QdrantClient(host=host, port=port)
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"✅ Qdrant connected. Found {len(collections.collections)} collections")
            
            # Mark as not using fallback
            self.use_fallback = False
            
            # Try to create collections, but don't fail if it doesn't work
            try:
                self._ensure_multi_modal_collections()
            except Exception as e:
                logger.warning(f"⚠️ Could not create collections: {e}")
                logger.info("   Using existing collections only")
            
        except ImportError as e:
            logger.warning(f"⚠️ Qdrant client not installed: {e}")
            logger.info("   Using in-memory fallback storage")
            logger.info("   Install: pip install qdrant-client")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Qdrant: {e}")
            logger.info("   Using in-memory fallback storage")
            logger.info("   Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
    
    def _ensure_multi_modal_collections(self):
        """Ensure multi-modal collections exist - SAFE VERSION."""
        if self.use_fallback or not self.client:
            return
        
        try:
            from qdrant_client.http import models
            
            collections = self.client.get_collections()
            existing_collections = [c.name for c in collections.collections]
            
            # Create collections for different modalities
            for modality, collection_name in self.multi_modal_collections.items():
                if collection_name not in existing_collections:
                    # Determine vector size based on modality
                    vector_size = 384  # Default for text
                    if modality == "vision":
                        vector_size = 512  # CLIP embeddings
                    elif modality == "multimodal":
                        vector_size = 896  # Concatenated text+vision
                    
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    logger.info(f"Created multi-modal collection: {collection_name} (dim={vector_size})")
                    
        except Exception as e:
            logger.warning(f"Failed to create multi-modal collections: {e}")
            # Don't switch to fallback, just use existing collections
    
    def store_multi_modal_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Store a multi-modal chunk in appropriate collection."""
        if self.use_fallback or not self.client:
            return self._fallback_store_chunk(chunk)
        
        try:
            from qdrant_client.http import models
            
            # Determine modality and collection
            modality = chunk.get("metadata", {}).get("modality", "text")
            collection_name = self.multi_modal_collections.get(modality, self.multi_modal_collections["text"])
            
            # Prepare embedding based on modality
            embedding = []
            if modality == "text":
                embedding = chunk.get("text_embedding", []) or chunk.get("embedding", [])
            elif modality == "vision":
                embedding = chunk.get("visual_embedding", []) or chunk.get("embedding", [])
            elif modality == "multimodal":
                # Concatenate text and vision embeddings
                text_embed = chunk.get("text_embedding", []) or []
                vision_embed = chunk.get("visual_embedding", []) or []
                embedding = text_embed + vision_embed
            
            # Ensure embedding has correct dimension
            expected_dim = {
                "text": 384,
                "vision": 512,
                "multimodal": 896
            }.get(modality, 384)
            
            if len(embedding) != expected_dim:
                logger.warning(f"Embedding dimension mismatch: {len(embedding)} != {expected_dim}")
                # Pad or truncate
                if len(embedding) > expected_dim:
                    embedding = embedding[:expected_dim]
                else:
                    embedding = embedding + [0.0] * (expected_dim - len(embedding))
            
            # Create point
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "document_id": chunk.get("metadata", {}).get("document_id", ""),
                    "text": chunk.get("text", ""),
                    "visual_description": chunk.get("visual_description", ""),
                    **chunk.get("metadata", {})
                }
            )
            
            # Store in appropriate collection
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored chunk in {collection_name} collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store multi-modal chunk in Qdrant: {e}")
            logger.info("Falling back to in-memory storage")
            return self._fallback_store_chunk(chunk)
    
    def search_multi_modal(self, query_embedding: List[float], modality: str = "multimodal",
                          document_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search across multi-modal collections."""
        if self.use_fallback or not self.client:
            return self._fallback_search(query_embedding, document_id, limit)
        
        try:
            from qdrant_client.http import models
            
            # Determine collection(s) to search
            collections_to_search = []
            if modality == "multimodal":
                # Search all collections
                collections_to_search = list(self.multi_modal_collections.values())
            else:
                # Search specific modality collection
                collection_name = self.multi_modal_collections.get(modality)
                if collection_name:
                    collections_to_search = [collection_name]
            
            all_results = []
            
            for collection_name in collections_to_search:
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
                
                # Search in collection
                search_result = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    query_filter=filter_condition,
                    limit=limit
                )
                
                # Add collection info to results
                for hit in search_result:
                    result = {
                        "score": hit.score,
                        "text": hit.payload.get("text", ""),
                        "visual_description": hit.payload.get("visual_description", ""),
                        "metadata": {k: v for k, v in hit.payload.items() 
                                   if k not in ["text", "visual_description"]},
                        "collection": collection_name,
                        "modality": self._get_modality_from_collection(collection_name)
                    }
                    all_results.append(result)
            
            # Sort all results by score
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Multi-modal search failed: {e}")
            logger.info("Falling back to in-memory search")
            return self._fallback_search(query_embedding, document_id, limit)
    
    def search_similar(self, query_embedding: List[float], document_id: Optional[str] = None, 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Simplified search interface for agents.py compatibility."""
        return self.search_multi_modal(query_embedding, "text", document_id, limit)
    
    def _get_modality_from_collection(self, collection_name: str) -> str:
        """Extract modality from collection name."""
        for modality, name in self.multi_modal_collections.items():
            if name == collection_name:
                return modality
        return "text"
    
    def _fallback_store_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Fallback storage for chunks."""
        if "fallback_storage" not in self.__dict__:
            self.fallback_storage = {}
        
        document_id = chunk.get("metadata", {}).get("document_id", "unknown")
        if document_id not in self.fallback_storage:
            self.fallback_storage[document_id] = []
        
        chunk_id = str(uuid.uuid4())
        chunk_with_id = {
            "id": chunk_id,
            **chunk,
            "document_id": document_id
        }
        self.fallback_storage[document_id].append(chunk_with_id)
        
        logger.info(f"📦 Stored chunk in fallback storage for {document_id}")
        return True
    
    def _fallback_search(self, query_embedding: List[float], 
                        document_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback search implementation."""
        if not hasattr(self, 'fallback_storage'):
            return []
        
        results = []
        
        for doc_id, chunks in self.fallback_storage.items():
            if document_id and doc_id != document_id:
                continue
            
            for chunk in chunks:
                # Get embedding from chunk
                chunk_embedding = (chunk.get("text_embedding") or 
                                 chunk.get("visual_embedding") or 
                                 chunk.get("embedding") or [])
                
                if not chunk_embedding:
                    continue
                
                # Calculate cosine similarity
                import numpy as np
                
                query_np = np.array(query_embedding)
                chunk_np = np.array(chunk_embedding)
                
                # Handle dimension mismatch
                min_dim = min(len(query_np), len(chunk_np))
                if min_dim == 0:
                    similarity = 0.0
                else:
                    query_slice = query_np[:min_dim]
                    chunk_slice = chunk_np[:min_dim]
                    
                    norm_query = np.linalg.norm(query_slice)
                    norm_chunk = np.linalg.norm(chunk_slice)
                    
                    if norm_query > 0 and norm_chunk > 0:
                        similarity = np.dot(query_slice, chunk_slice) / (norm_query * norm_chunk)
                    else:
                        similarity = 0.0
                
                results.append({
                    "score": float(similarity),
                    "text": chunk.get("text", ""),
                    "visual_description": chunk.get("visual_description", ""),
                    "metadata": chunk.get("metadata", {}),
                    "document_id": doc_id,
                    "collection": "fallback",
                    "modality": chunk.get("metadata", {}).get("modality", "text")
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_collection_info(self):
        """Get information about collections."""
        if self.use_fallback or not self.client:
            return {
                "status": "using_fallback",
                "stored_chunks": sum(len(chunks) for chunks in self.fallback_storage.values())
            }
        
        try:
            collections = self.client.get_collections()
            return {
                "status": "connected",
                "collections": [c.name for c in collections.collections],
                "multi_modal_collections": self.multi_modal_collections
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global instance
qdrant_store = QdrantStore()
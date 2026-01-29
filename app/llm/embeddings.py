"""
Embedding generation for multi-modal RAG - SAFE VERSION.
"""
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text and layout-aware chunks - SAFE VERSION."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model - SAFELY."""
        self.model = None
        self.model_name = model_name
        self.dimension = 384  # Default dimension for fallback
        
        logger.info(f"🔄 Initializing EmbeddingGenerator with model: {model_name}")
        self._initialize_model_safe()
    
    def _initialize_model_safe(self):
        """Initialize embedding model - DON'T CRASH ON FAILURE."""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            
            # Load model
            self.model = SentenceTransformer(self.model_name)
            
            # Get dimension
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Embedding model loaded: {self.model_name}")
            logger.info(f"   Embedding dimension: {self.dimension}")
            
        except ImportError as e:
            logger.warning(f"⚠️ sentence-transformers not installed: {e}")
            logger.info("   Embeddings will return zero vectors")
            logger.info("   Install: pip install sentence-transformers")
            self.model = None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load embedding model: {e}")
            logger.info("   Embeddings will return zero vectors")
            self.model = None
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text - SAFELY."""
        if not text:
            logger.debug("Empty text provided for embedding")
            return [0.0] * self.dimension
        
        if self.model is None:
            logger.debug("⚠️ Embedding model not loaded, returning zero vector")
            # Return zero vector if model not available
            return [0.0] * self.dimension
        
        try:
            # Clean text
            text = str(text).strip()
            if len(text) < 3:
                logger.debug(f"Text too short for embedding: '{text}'")
                return [0.0] * self.dimension
                
            embedding = self.model.encode(text)
            embedding_list = embedding.tolist()
            
            # Validate embedding
            actual_dim = len(embedding_list)
            if actual_dim != self.dimension:
                logger.warning(f"Unexpected embedding dimension: {actual_dim} != {self.dimension}")
                # Pad or truncate to expected dimension
                if actual_dim > self.dimension:
                    embedding_list = embedding_list[:self.dimension]
                else:
                    embedding_list = embedding_list + [0.0] * (self.dimension - actual_dim)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Text embedding failed: {e}")
            # Return zero vector on error
            return [0.0] * self.dimension
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch - SAFELY."""
        if not texts:
            return []
        
        if self.model is None:
            logger.warning("Embedding model not available for batch")
            return [[0.0] * self.dimension for _ in texts]
        
        try:
            # Filter out empty texts
            valid_texts = [t for t in texts if t and len(str(t).strip()) >= 3]
            if not valid_texts:
                return [[0.0] * self.dimension for _ in texts]
            
            embeddings = self.model.encode(valid_texts)
            embeddings_list = embeddings.tolist()
            
            # Handle dimension mismatch if any
            result = []
            for emb in embeddings_list:
                if len(emb) != self.dimension:
                    if len(emb) > self.dimension:
                        emb = emb[:self.dimension]
                    else:
                        emb = emb + [0.0] * (self.dimension - len(emb))
                result.append(emb)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            return [[0.0] * self.dimension for _ in texts]
    
    def get_info(self) -> dict:
        """Get information about the embedding generator."""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "dimension": self.dimension,
            "status": "ready" if self.model else "fallback_mode"
        }


# Global instance
embedding_generator = EmbeddingGenerator()
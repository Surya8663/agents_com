"""
Embedding generation for multi-modal RAG.
"""
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text and layout-aware chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model."""
        self.model = None
        self.model_name = model_name
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Embedding model loaded: {self.model_name}")
            logger.info(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except ImportError as e:
            logger.error(f"❌ sentence-transformers not installed: {e}")
            logger.error("   Please install: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.model = None
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not text:
            logger.warning("Empty text provided for embedding")
            return [0.0] * 384
        
        if self.model is None:
            logger.error("❌ Embedding model not loaded")
            # Return zero vector if model not available
            return [0.0] * 384
        
        try:
            # Clean text
            text = str(text).strip()
            if len(text) < 3:
                return [0.0] * 384
                
            embedding = self.model.encode(text)
            embedding_list = embedding.tolist()
            
            # Validate embedding
            if len(embedding_list) != 384:
                logger.warning(f"Unexpected embedding dimension: {len(embedding_list)}")
                # Pad or truncate to 384
                if len(embedding_list) > 384:
                    embedding_list = embedding_list[:384]
                else:
                    embedding_list = embedding_list + [0.0] * (384 - len(embedding_list))
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Text embedding failed: {e}")
            # Return zero vector on error
            return [0.0] * 384
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch."""
        if not texts:
            return []
        
        if self.model is None:
            logger.error("Embedding model not available for batch")
            return [[0.0] * 384 for _ in texts]
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            return [[0.0] * 384 for _ in texts]


# Global instance
embedding_generator = EmbeddingGenerator()
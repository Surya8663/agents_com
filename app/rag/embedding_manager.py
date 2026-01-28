"""
Multi-modal embedding manager using REAL models.
"""
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image
import io

logger = logging.getLogger(__name__)


class MultiModalEmbeddingManager:
    """Manages REAL embeddings for text and vision."""
    
    def __init__(self):
        """Initialize REAL embedding models."""
        self.text_model = None
        self.vision_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize REAL embedding models."""
        try:
            # Text embedding model - Sentence Transformers
            from sentence_transformers import SentenceTransformer
            logger.info("🔄 Loading REAL text embedding model: all-MiniLM-L6-v2")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(f"✅ Text embedding model loaded (dim={self.text_model.get_sentence_embedding_dimension()})")
            
        except ImportError as e:
            logger.error(f"❌ Failed to load text embedding model: {e}")
            raise RuntimeError(f"REAL text embedding model required: {e}")
        
        try:
            # Vision embedding model - CLIP for cross-modal
            import clip
            import torch
            logger.info("🔄 Loading REAL vision embedding model: CLIP")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load CLIP model
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.vision_device = device
            logger.info("✅ CLIP vision embedding model loaded")
            
        except ImportError as e:
            logger.error(f"❌ Failed to load vision embedding model: {e}")
            logger.warning("⚠️ Vision embeddings will be text-based only")
            self.clip_model = None
    
    def embed_text(self, text: str) -> List[float]:
        """Generate REAL text embedding."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * 384
        
        try:
            # Clean and embed
            clean_text = str(text).strip()
            embedding = self.text_model.encode(clean_text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Text embedding failed: {e}")
            # Return zero vector but log error
            return [0.0] * 384
    
    def embed_vision(self, image: np.ndarray, description: Optional[str] = None) -> List[float]:
        """Generate REAL vision embedding."""
        if self.clip_model is None:
            logger.warning("CLIP model not available, using text embedding for vision")
            if description:
                return self.embed_text(description)
            return [0.0] * 512
        
        try:
            import torch
            from PIL import Image
            
            # Convert numpy array to PIL Image
            if len(image.shape) == 2:
                # Grayscale to RGB
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = Image.fromarray(image)
            
            # Preprocess for CLIP
            preprocessed = self.clip_preprocess(pil_image).unsqueeze(0).to(self.vision_device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed)
                embedding = image_features.cpu().numpy().flatten().tolist()
            
            logger.debug(f"Generated vision embedding: {len(embedding)} dim")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Vision embedding failed: {e}")
            # Fallback to text description if available
            if description:
                return self.embed_text(description)
            return [0.0] * 512
    
    def embed_multimodal(self, text: str, image: Optional[np.ndarray] = None, 
                        description: Optional[str] = None) -> Dict[str, List[float]]:
        """Generate multi-modal embeddings."""
        embeddings = {
            "text_embedding": self.embed_text(text),
            "visual_embedding": None
        }
        
        # Generate vision embedding if image is provided
        if image is not None and image.size > 0:
            embeddings["visual_embedding"] = self.embed_vision(image, description)
        elif description:
            # Use description for vision embedding
            embeddings["visual_embedding"] = self.embed_text(description)
        
        return embeddings
    
    def batch_embed_text(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts."""
        if not texts:
            return []
        
        try:
            embeddings = self.text_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Batch text embedding failed: {e}")
            return [[0.0] * 384 for _ in texts]
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        try:
            a = np.array(embedding1)
            b = np.array(embedding2)
            
            # Normalize
            a_norm = a / np.linalg.norm(a)
            b_norm = b / np.linalg.norm(b)
            
            # Cosine similarity
            similarity = np.dot(a_norm, b_norm)
            return float(similarity)
        except Exception as e:
            logger.error(f"❌ Similarity computation failed: {e}")
            return 0.0


# Global instance
embedding_manager = MultiModalEmbeddingManager()
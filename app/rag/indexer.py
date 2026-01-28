"""
Multi-modal indexer for RAG system.
"""
import logging
import uuid
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.rag.multimodal_schema import RAGChunk, Modality
from app.rag.embedding_manager import embedding_manager
from app.vectorstore.qdrant_client import qdrant_store
from app.config.settings import settings

logger = logging.getLogger(__name__)


class MultiModalIndexer:
    """Indexes multi-modal document chunks."""
    
    def __init__(self):
        """Initialize indexer."""
        self.embedding_manager = embedding_manager
    
    def create_chunks_from_agent_results(self, agent_results: Dict[str, Any]) -> List[RAGChunk]:
        """Create RAG chunks from agent pipeline results."""
        chunks = []
        document_id = agent_results.get("document_id", "")
        
        # Extract from fused document
        fused_doc = agent_results.get("fused_document", {})
        text_analysis = agent_results.get("text_analysis", {})
        vision_analysis = agent_results.get("vision_analysis", {})
        layout_data = agent_results.get("layout_data", {})
        
        # 1. Text chunks from key-value pairs
        key_value_pairs = text_analysis.get("key_value_pairs", {})
        for key, value in key_value_pairs.items():
            chunk = RAGChunk(
                id=f"text_{uuid.uuid4().hex[:8]}",
                document_id=document_id,
                page_number=1,  # Default, will be refined
                text=f"{key}: {value}",
                modality=Modality.TEXT,
                agent_source="text_agent",
                confidence=text_analysis.get("semantic_confidence", 0.7),
                metadata={"field_type": "key_value_pair", "key": key}
            )
            chunks.append(chunk)
        
        # 2. Text chunks from OCR regions
        ocr_data = agent_results.get("ocr_data", {})
        if "pages" in ocr_data:
            for page in ocr_data["pages"]:
                page_num = page.get("page_number", 1)
                for region in page.get("regions", []):
                    ocr_text = region.get("ocr_text", "")
                    if ocr_text and len(ocr_text.strip()) > 0:
                        chunk = RAGChunk(
                            id=f"ocr_{uuid.uuid4().hex[:8]}",
                            document_id=document_id,
                            page_number=page_num,
                            text=ocr_text,
                            modality=Modality.TEXT,
                            agent_source="ocr_engine",
                            confidence=region.get("ocr_confidence", 0.5),
                            metadata={
                                "region_type": region.get("type", "text_block"),
                                "bbox": region.get("bbox", {}),
                                "engine": "easyocr"
                            }
                        )
                        chunks.append(chunk)
        
        # 3. Vision chunks from layout analysis
        if "pages" in layout_data:
            for page in layout_data["pages"]:
                page_num = page.get("page_number", 1)
                for detection in page.get("detections", []):
                    chunk = RAGChunk(
                        id=f"vision_{uuid.uuid4().hex[:8]}",
                        document_id=document_id,
                        page_number=page_num,
                        visual_description=f"{detection.get('type', 'element')} at position ({detection.get('bbox', {}).get('x1', 0):.2f}, {detection.get('bbox', {}).get('y1', 0):.2f})",
                        bbox=detection.get("bbox"),
                        region_type=detection.get("type", "unknown"),
                        modality=Modality.VISION,
                        agent_source="vision_agent",
                        confidence=detection.get("confidence", 0.6),
                        metadata={
                            "detection_type": detection.get("type"),
                            "page_dimensions": f"{page.get('page_width', 0)}x{page.get('page_height', 0)}"
                        }
                    )
                    chunks.append(chunk)
        
        # 4. Fusion chunks from fused extractions
        fused_extractions = fused_doc.get("fused_extractions", {})
        for key, value_info in fused_extractions.items():
            if isinstance(value_info, dict):
                value = value_info.get("value", "")
                confidence = value_info.get("multi_modal_confidence", 0.8)
            else:
                value = value_info
                confidence = fused_doc.get("fusion_confidence", 0.8)
            
            chunk = RAGChunk(
                id=f"fusion_{uuid.uuid4().hex[:8]}",
                document_id=document_id,
                page_number=1,  # Will be refined
                text=f"{key}: {value}",
                modality=Modality.MULTIMODAL,
                agent_source="fusion_agent",
                confidence=confidence,
                metadata={
                    "extraction_type": "fused",
                    "source": "multi_modal_fusion"
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} RAG chunks from document {document_id}")
        return chunks
    
    def index_chunks(self, chunks: List[RAGChunk]) -> bool:
        """Index chunks with embeddings in vector store."""
        if not chunks:
            logger.warning("No chunks to index")
            return False
        
        try:
            # Generate embeddings for each chunk
            for chunk in chunks:
                # Generate text embedding
                if chunk.text:
                    chunk.text_embedding = self.embedding_manager.embed_text(chunk.text)
                
                # Generate visual embedding
                if chunk.visual_description:
                    chunk.visual_embedding = self.embedding_manager.embed_text(chunk.visual_description)
                elif chunk.text and not chunk.visual_description:
                    # Use text as visual description fallback
                    chunk.visual_embedding = chunk.text_embedding
            
            # Prepare chunks for storage
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = chunk.dict()
                
                # Use text embedding as primary vector
                primary_embedding = chunk.text_embedding or chunk.visual_embedding or [0.0] * 384
                
                chunk_dicts.append({
                    "embedding": primary_embedding,
                    "text": chunk.text or "",
                    "visual_description": chunk.visual_description or "",
                    "metadata": {
                        "document_id": chunk.document_id,
                        "page_number": chunk.page_number,
                        "modality": chunk.modality,
                        "agent_source": chunk.agent_source,
                        "confidence": chunk.confidence,
                        "region_type": chunk.region_type or "",
                        "bbox": chunk.bbox.dict() if chunk.bbox else {},
                        **chunk.metadata
                    }
                })
            
            # Store in vector database
            document_id = chunks[0].document_id if chunks else ""
            success = qdrant_store.store_document_chunks(document_id, chunk_dicts)
            
            if success:
                logger.info(f"✅ Indexed {len(chunks)} chunks for document {document_id}")
                return True
            else:
                logger.error(f"❌ Failed to index chunks for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            return False
    
    def index_document(self, document_id: str) -> bool:
        """Index an entire document from stored agent results."""
        try:
            # Load agent results
            doc_dir = settings.DOCUMENTS_DIR / document_id
            agent_file = doc_dir / "agents" / "agent_results.json"
            
            if not agent_file.exists():
                logger.error(f"No agent results found for document {document_id}")
                return False
            
            with open(agent_file, "r", encoding="utf-8") as f:
                agent_results = json.load(f)
            
            # Create chunks
            chunks = self.create_chunks_from_agent_results(agent_results)
            
            # Index chunks
            return self.index_chunks(chunks)
            
        except Exception as e:
            logger.error(f"❌ Document indexing failed: {e}")
            return False


# Global instance
indexer = MultiModalIndexer()
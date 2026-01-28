"""
Multi-modal RAG data schemas.
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class Modality(str):
    """Modality types."""
    TEXT = "text"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class BoundingBox(BaseModel):
    """Normalized bounding box."""
    x1: float = Field(..., ge=0, le=1, description="Top-left x (normalized)")
    y1: float = Field(..., ge=0, le=1, description="Top-left y (normalized)")
    x2: float = Field(..., ge=0, le=1, description="Bottom-right x (normalized)")
    y2: float = Field(..., ge=0, le=1, description="Bottom-right y (normalized)")
    
    def area(self) -> float:
        """Calculate area of bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class RAGChunk(BaseModel):
    """Base RAG chunk with multi-modal support."""
    id: str = Field(..., description="Unique chunk ID")
    document_id: str = Field(..., description="Document identifier")
    page_number: int = Field(..., description="Page number (1-indexed)")
    
    # Content
    text: Optional[str] = Field(None, description="Text content")
    visual_description: Optional[str] = Field(None, description="Visual description")
    
    # Spatial context
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box on page")
    region_type: Optional[str] = Field(None, description="Region type (text_block, table, etc.)")
    
    # Modality
    modality: Modality = Field(..., description="Content modality")
    
    # Source tracking
    agent_source: str = Field(..., description="Source agent (vision_agent, text_agent, etc.)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    
    # Embeddings
    text_embedding: Optional[List[float]] = Field(None, description="Text embedding vector")
    visual_embedding: Optional[List[float]] = Field(None, description="Visual embedding vector")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chunk_001",
                "document_id": "doc_123",
                "page_number": 1,
                "text": "Invoice number: INV-2023-001",
                "visual_description": "Blue header with company logo on right",
                "bbox": {"x1": 0.1, "y1": 0.05, "x2": 0.9, "y2": 0.15},
                "region_type": "header",
                "modality": "multimodal",
                "agent_source": "fusion_agent",
                "confidence": 0.92,
                "text_embedding": [0.1, 0.2, ...],
                "visual_embedding": [0.3, 0.4, ...],
                "metadata": {"font_size": 14, "color": "blue"}
            }
        }


class RAGQuery(BaseModel):
    """RAG query with multi-modal support."""
    query: str = Field(..., description="Natural language query")
    modality: Optional[Modality] = Field(None, description="Preferred modality")
    document_id: Optional[str] = Field(None, description="Filter by document")
    page_number: Optional[int] = Field(None, description="Filter by page")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    similarity_threshold: float = Field(0.7, ge=0, le=1, description="Minimum similarity")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the invoice total amount?",
                "modality": "multimodal",
                "document_id": "doc_123",
                "top_k": 5,
                "similarity_threshold": 0.8
            }
        }


class RAGResult(BaseModel):
    """RAG retrieval result."""
    chunk: RAGChunk
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")
    evidence_type: str = Field(..., description="Type of evidence")
    
    class Config:
        schema_extra = {
            "example": {
                "chunk": {...},
                "similarity_score": 0.89,
                "relevance_score": 0.92,
                "evidence_type": "text_and_visual"
            }
        }


class RAGResponse(BaseModel):
    """Complete RAG response."""
    query: str
    results: List[RAGResult] = Field(default_factory=list)
    answer: Optional[str] = Field(None, description="Generated answer")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")
    traceability: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the invoice total amount?",
                "results": [...],
                "answer": "The invoice total amount is $1,250.00",
                "confidence": 0.88,
                "traceability": {
                    "sources_used": 3,
                    "pages_referenced": [1, 2],
                    "modalities": ["text", "vision"]
                }
            }
        }
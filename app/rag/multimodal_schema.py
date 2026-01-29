"""
Multi-modal RAG data schemas.
"""
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class Modality(str, Enum):
    """Modality types."""
    TEXT = "text"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class BoundingBox(BaseModel):
    """Normalized bounding box."""
    model_config = ConfigDict(extra='forbid')
    
    x1: float = Field(..., ge=0, le=1, description="Top-left x (normalized)")
    y1: float = Field(..., ge=0, le=1, description="Top-left y (normalized)")
    x2: float = Field(..., ge=0, le=1, description="Bottom-right x (normalized)")
    y2: float = Field(..., ge=0, le=1, description="Bottom-right y (normalized)")
    
    def area(self) -> float:
        """Calculate area of bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class RAGChunk(BaseModel):
    """Base RAG chunk with multi-modal support."""
    model_config = ConfigDict(extra='allow')
    
    id: str = Field(..., description="Unique chunk ID")
    document_id: str = Field(..., description="Document identifier")
    page_number: int = Field(..., description="Page number (1-indexed)")
    
    # Content
    text: Optional[str] = Field(None, description="Text content")
    visual_description: Optional[str] = Field(None, description="Visual description")
    
    # Spatial context
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box on page")
    region_type: Optional[str] = Field(None, description="Region type (text_block, table, etc.)")
    
    # Modality - use the Enum type directly
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


class RAGQuery(BaseModel):
    """RAG query with multi-modal support."""
    model_config = ConfigDict(extra='forbid')
    
    query: str = Field(..., description="Natural language query")
    modality: Optional[Modality] = Field(None, description="Preferred modality")
    document_id: Optional[str] = Field(None, description="Filter by document")
    page_number: Optional[int] = Field(None, description="Filter by page")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    similarity_threshold: float = Field(0.7, ge=0, le=1, description="Minimum similarity")


class RAGResult(BaseModel):
    """RAG retrieval result."""
    model_config = ConfigDict(extra='forbid')
    
    chunk: RAGChunk
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")
    evidence_type: str = Field(..., description="Type of evidence")


class RAGResponse(BaseModel):
    """Complete RAG response."""
    model_config = ConfigDict(extra='forbid')
    
    query: str
    results: List[RAGResult] = Field(default_factory=list)
    answer: Optional[str] = Field(None, description="Generated answer")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")
    traceability: Dict[str, Any] = Field(default_factory=dict)
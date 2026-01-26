"""
OCR data models for Phase 3.
"""
import uuid
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class OCRBoundingBox(BaseModel):
    """Normalized bounding box for OCR results."""
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)


class OCRRegionResult(BaseModel):
    """OCR result for a single region."""
    region_id: str = Field(...)
    type: Literal["text_block", "table", "figure", "signature"] = Field(...)
    bbox: OCRBoundingBox = Field(...)
    ocr_text: Optional[str] = Field(None)
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    engine: str = Field(default="PaddleOCR")
    language: str = Field(default="en")
    word_boxes: Optional[List[Dict[str, Any]]] = Field(None)
    ocr_processed: bool = Field(default=False)
    ocr_skipped_reason: Optional[str] = Field(None)


class PageOCRResult(BaseModel):
    """OCR results for a single page."""
    page_number: int = Field(..., ge=1)
    document_id: uuid.UUID = Field(...)
    image_path: str = Field(...)
    image_width: int = Field(..., ge=1)
    image_height: int = Field(..., ge=1)
    regions: List[OCRRegionResult] = Field(default_factory=list)
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    total_regions: int = Field(default=0)
    ocr_regions: int = Field(default=0)
    skipped_regions: int = Field(default=0)
    average_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class DocumentOCRResult(BaseModel):
    """Complete document OCR results."""
    document_id: uuid.UUID = Field(...)
    pages: List[PageOCRResult] = Field(default_factory=list)
    total_pages: int = Field(default=0)
    total_regions: int = Field(default=0)
    ocr_regions: int = Field(default=0)
    skipped_regions: int = Field(default=0)
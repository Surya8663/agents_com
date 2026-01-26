# app/vision/schema.py
import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates (0-1 range)."""
    x1: float = Field(..., ge=0.0, le=1.0, description="Normalized left coordinate")
    y1: float = Field(..., ge=0.0, le=1.0, description="Normalized top coordinate")
    x2: float = Field(..., ge=0.0, le=1.0, description="Normalized right coordinate")
    y2: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom coordinate")
    
    @classmethod
    def from_pixel_coords(cls, x1: int, y1: int, x2: int, y2: int, 
                          image_width: int, image_height: int) -> "BoundingBox":
        """Convert pixel coordinates to normalized coordinates."""
        return cls(
            x1=x1 / image_width,
            y1=y1 / image_height,
            x2=x2 / image_width,
            y2=y2 / image_height
        )
    
    def to_pixel_coords(self, image_width: int, image_height: int) -> tuple:
        """Convert normalized coordinates back to pixel coordinates."""
        return (
            int(self.x1 * image_width),
            int(self.y1 * image_height),
            int(self.x2 * image_width),
            int(self.y2 * image_height)
        )
    
    def area(self) -> float:
        """Calculate area of bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection coordinates
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        
        # Calculate intersection area
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height
        
        # Calculate union area
        union_area = self.area() + other.area() - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0


class Detection(BaseModel):
    """Single layout element detection."""
    type: Literal["text_block", "table", "figure", "signature"] = Field(
        ..., description="Type of detected element"
    )
    bbox: BoundingBox = Field(..., description="Normalized bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    page_width: int = Field(..., ge=1, description="Original image width in pixels")
    page_height: int = Field(..., ge=1, description="Original image height in pixels")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PageLayout(BaseModel):
    """Layout analysis results for a single page."""
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    image_path: str = Field(..., description="Path to the page image file")
    page_width: int = Field(..., ge=1, description="Image width in pixels")
    page_height: int = Field(..., ge=1, description="Image height in pixels")
    detections: List[Detection] = Field(default_factory=list, description="List of detected elements")
    
    model_config = ConfigDict(
        json_encoders={
            np.float32: lambda x: float(x),
            np.float64: lambda x: float(x)
        }
    )


class DocumentLayout(BaseModel):
    """Complete document layout analysis results."""
    document_id: uuid.UUID = Field(..., description="Document identifier from Phase 1")
    pages: List[PageLayout] = Field(default_factory=list, description="Layout analysis per page")
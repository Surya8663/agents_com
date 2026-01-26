import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class PageMetadata(BaseModel):
    """Metadata for a single page."""
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    image_path: str = Field(..., description="Path to the page image file")
    native_text_available: bool = Field(..., description="Whether native text was extracted")
    native_text_length: int = Field(..., ge=0, description="Length of extracted text if available")
    native_text_content: Optional[str] = Field(None, description="Actual text content if available")


class DocumentMetadata(BaseModel):
    """Complete document metadata."""
    document_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., ge=1, description="Total number of pages in document")
    upload_timestamp: datetime = Field(default_factory=datetime.now, description="Document upload timestamp")
    file_size_bytes: int = Field(..., ge=1, description="File size in bytes")
    pages: List[PageMetadata] = Field(default_factory=list, description="Page metadata list")


class DocumentUploadResponse(BaseModel):
    """API response for document upload."""
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Status message")
    document_id: uuid.UUID = Field(..., description="Assigned document ID")
    total_pages: int = Field(..., ge=1, description="Number of pages processed")
    pages_processed: int = Field(..., ge=1, description="Pages successfully processed")
    metadata_path: str = Field(..., description="Path to metadata file")

    # Pydantic v2 configuration - add model_config as a class variable
    model_config = ConfigDict(
        json_encoders={
            uuid.UUID: str,
            datetime: lambda dt: dt.isoformat()
        },
        from_attributes=True
    )
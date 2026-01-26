import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.ingestion.schema import DocumentMetadata, DocumentUploadResponse
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.page_processor import PageProcessor
from app.utils.validators import validate_pdf_file, validate_file_corruption
from app.utils.file_utils import save_uploaded_file, save_metadata

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    
    This endpoint:
    1. Validates the PDF file
    2. Creates a unique document ID
    3. Saves the original file
    4. Converts each page to an image
    5. Extracts native text if available
    6. Generates comprehensive metadata
    """
    # 1. Validate file
    is_valid, error_msg = validate_pdf_file(file)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File validation failed: {error_msg}"
        )
    
    # Generate document ID
    document_id = uuid.uuid4()
    
    try:
        # 2. Save uploaded file
        original_path, relative_path = save_uploaded_file(file, document_id)
        
        # 3. Validate PDF structure and corruption
        is_valid_corruption, corruption_msg = validate_file_corruption(original_path)
        if not is_valid_corruption:
            # Clean up the saved file
            shutil.rmtree(settings.DOCUMENTS_DIR / str(document_id), ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PDF validation failed: {corruption_msg}"
            )
        
        # 4. Get page count
        pdf_loader = PDFLoader()
        total_pages = pdf_loader.get_page_count(original_path)
        
        # 5. Process all pages
        processor = PageProcessor(original_path, str(document_id))
        pages_metadata = processor.process_all_pages(total_pages)
        
        # 6. Create complete metadata
        document_metadata = DocumentMetadata(
            filename=file.filename,
            total_pages=total_pages,
            file_size_bytes=original_path.stat().st_size,
            pages=pages_metadata
        )
        
        # 7. Save metadata to file
        metadata_path = save_metadata(document_metadata, document_id)
        
        # 8. Return response
        return DocumentUploadResponse(
            success=True,
            message=f"Document processed successfully: {total_pages} pages",
            document_id=document_id,
            total_pages=total_pages,
            pages_processed=len(pages_metadata),
            metadata_path=str(metadata_path.relative_to(settings.BASE_DATA_DIR))
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        import shutil
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.get("/status/{document_id}")
async def get_processing_status(document_id: uuid.UUID):
    """Check processing status of a document."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    metadata_path = doc_dir / "metadata.json"
    
    if not doc_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    if metadata_path.exists():
        return {"status": "completed", "document_id": document_id}
    elif (doc_dir / "original.pdf").exists():
        return {"status": "processing", "document_id": document_id}
    else:
        return {"status": "failed", "document_id": document_id}
import os
from pathlib import Path
from typing import Tuple
from fastapi import HTTPException, UploadFile

from app.config.settings import settings


def validate_pdf_file(file: UploadFile) -> Tuple[bool, str]:
    """
    Validate PDF file.
    Returns (is_valid, error_message)
    """
    # Check file extension
    if not file.filename.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    # Check content type
    allowed_content_types = ['application/pdf', 'application/x-pdf']
    if file.content_type not in allowed_content_types:
        return False, f"Invalid content type: {file.content_type}"
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_FILE_SIZE_BYTES:
        return False, f"File size exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    # Check file magic number using filetype (Windows compatible)
    try:
        import filetype
        file_content = file.file.read(1024)
        file.file.seek(0)
        
        kind = filetype.guess(file_content)
        if kind is None:
            # Couldn't determine file type, rely on extension
            print("Warning: Could not determine file type from content")
        elif kind.mime not in allowed_content_types:
            return False, f"File content does not match PDF format. Detected: {kind.mime}"
            
    except ImportError:
        # If filetype is not available, use simple PDF header check
        file_content = file.file.read(5)
        file.file.seek(0)
        
        # Check for PDF header: %PDF-
        if file_content[:4] != b'%PDF':
            # Some PDFs might have version after header, check longer
            file.file.seek(0)
            longer_content = file.file.read(1024)
            file.file.seek(0)
            if b'%PDF' not in longer_content:
                return False, "File does not appear to be a valid PDF (missing PDF header)"
    
    return True, ""


def validate_file_corruption(file_path: Path) -> Tuple[bool, str]:
    """
    Check if PDF file is corrupted.
    This is a basic check - more thorough validation would be needed for production.
    """
    try:
        # Try PyPDF2
        import PyPDF2
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            # Access pages to trigger any parsing errors
            _ = len(pdf_reader.pages)
            # Try to read some metadata
            _ = pdf_reader.metadata if hasattr(pdf_reader, 'metadata') else {}
        
        return True, ""
    except ImportError:
        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                _ = len(pdf.pages)
            return True, ""
        except Exception as e:
            return False, f"PDF validation failed: {str(e)}"
    except Exception as e:
        return False, f"PDF appears to be corrupted or malformed: {str(e)}"
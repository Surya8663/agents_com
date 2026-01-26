import fitz  # PyMuPDF for basic PDF validation
import pdfplumber
from pathlib import Path
from typing import Tuple, Optional

from app.config.settings import settings


class PDFLoader:
    """Handles PDF loading and basic validation."""
    
    @staticmethod
    def get_page_count(pdf_path: Path) -> int:
        """Get total number of pages in PDF."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {str(e)}")
    
    @staticmethod
    def extract_native_text(pdf_path: Path, page_num: int) -> Optional[str]:
        """
        Extract native text from a specific page.
        Returns None if no text is available.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < 1 or page_num > len(pdf.pages):
                    raise ValueError(f"Page number {page_num} out of range")
                
                page = pdf.pages[page_num - 1]  # pdfplumber uses 0-index
                text = page.extract_text()
                
                return text if text and text.strip() else None
        except Exception as e:
            # Log but don't fail - scanned PDFs won't have native text
            return None
    
    @staticmethod
    def validate_pdf_structure(pdf_path: Path) -> Tuple[bool, str]:
        """Validate PDF structure using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            
            if page_count == 0:
                return False, "PDF has no pages"
            
            # Check if document is encrypted
            if doc.is_encrypted:
                return False, "PDF is encrypted and cannot be processed"
            
            doc.close()
            return True, ""
        except Exception as e:
            return False, f"PDF structure validation failed: {str(e)}"
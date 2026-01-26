"""
Phase 3 OCR Module
"""
# Import everything - OCRManager has built-in fallback
from app.ocr.engine import OCRManager
from app.ocr.schema import OCRBoundingBox, OCRRegionResult, PageOCRResult

__all__ = [
    'OCRManager',
    'OCRBoundingBox',
    'OCRRegionResult',
    'PageOCRResult'
]
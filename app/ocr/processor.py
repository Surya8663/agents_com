# app/ocr/processor.py - UPDATED FOR EasyOCR
"""
Main OCR processor for Phase 3 - REAL EasyOCR only.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from app.config.settings import settings
from app.ocr.engine import OCRManager
from app.ocr.region_cropper import RegionCropper
from app.ocr.schema import PageOCRResult, OCRRegionResult, OCRBoundingBox

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Processes REAL OCR for documents - NO MOCK DATA."""
    
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_manager = OCRManager(lang, use_gpu)
        self.region_cropper = RegionCropper()
        
        if self.ocr_manager.mode != "real":
            logger.error("❌ OCR is in FALLBACK mode - NOT REAL OCR!")
            raise RuntimeError("EasyOCR failed to initialize. REAL OCR required.")
        
        logger.info(f"✅ REAL EasyOCR Processor initialized (lang={lang}, GPU={use_gpu})")
    
    def _validate_real_text(self, text: Optional[str]) -> bool:
        """Validate that text is REAL, not mock data."""
        if not text:
            return False
        
        # Common mock phrases from your old fallback
        mock_phrases = [
            "INVOICE #INV-", 
            "PURCHASE ORDER",
            "CONTRACT AGREEMENT", 
            "PROJECT REPORT",
            "FINANCIAL STATEMENT",
            "MEMORANDUM",
            "LETTER OF ACCEPTANCE",
            "TECHNICAL SPECIFICATION"
        ]
        
        # Check if text contains mock patterns
        for phrase in mock_phrases:
            if phrase in text:
                logger.warning(f"⚠️ Mock data detected: '{phrase}'")
                return False
        
        return True
    
    def process_page(self, document_id: uuid.UUID, page_data: Dict[str, Any]) -> Optional[PageOCRResult]:
        """Process REAL OCR for a single page."""
        try:
            # Extract page data
            page_number = page_data.get("page_number", 1)
            image_path = page_data.get("image_path", "")
            page_width = page_data.get("page_width", 0)
            page_height = page_data.get("page_height", 0)
            detections = page_data.get("detections", [])
            
            if not image_path:
                logger.error(f"No image path for page {page_number}")
                return None
            
            # Get absolute image path
            doc_dir = settings.DOCUMENTS_DIR / str(document_id)
            abs_image_path = doc_dir / image_path
            
            if not abs_image_path.exists():
                abs_image_path = settings.BASE_DATA_DIR / image_path
                if not abs_image_path.exists():
                    logger.error(f"Image not found: {image_path}")
                    return None
            
            # Crop regions
            cropped_regions = self.region_cropper.crop_regions_from_page(abs_image_path, detections)
            
            if not cropped_regions:
                logger.warning(f"No regions cropped for page {page_number}")
                cropped_regions = []
            
            # Process with REAL OCR
            ocr_results = self.ocr_manager.process_regions(cropped_regions)
            
            # Convert and validate REAL OCR results
            region_results = []
            real_ocr_count = 0
            empty_ocr_count = 0
            confidences = []
            
            for result in ocr_results:
                detection = result["detection"]
                detection_type = detection.get("type", "text_block")
                bbox_dict = detection.get("bbox", {})
                
                # Create bbox
                ocr_bbox = OCRBoundingBox(
                    x1=float(bbox_dict.get("x1", 0)),
                    y1=float(bbox_dict.get("y1", 0)),
                    x2=float(bbox_dict.get("x2", 0)),
                    y2=float(bbox_dict.get("y2", 0))
                )
                
                # Get OCR results
                ocr_text = result.get("ocr_text", "")
                ocr_confidence = result.get("ocr_confidence", 0.0)
                
                # Validate it's REAL text
                is_real_text = self._validate_real_text(ocr_text) and len(str(ocr_text).strip()) > 0
                
                # Create region result
                region_result = OCRRegionResult(
                    region_id=result["region_id"],
                    type=detection_type,
                    bbox=ocr_bbox,
                    ocr_text=ocr_text if is_real_text else "",
                    ocr_confidence=ocr_confidence if is_real_text else 0.0,
                    engine="easyocr_real",  # CHANGED from paddleocr_real
                    language=self.lang,
                    word_boxes=result.get("word_boxes", []),
                    ocr_processed=is_real_text,
                    ocr_skipped_reason="" if is_real_text else "No real text extracted"
                )
                
                region_results.append(region_result)
                
                # Track statistics
                if is_real_text:
                    real_ocr_count += 1
                    if ocr_confidence:
                        confidences.append(ocr_confidence)
                else:
                    empty_ocr_count += 1
            
            # Calculate average confidence
            avg_confidence = None
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
            
            # Create page result
            page_result = PageOCRResult(
                page_number=page_number,
                document_id=document_id,
                image_path=image_path,
                image_width=page_width,
                image_height=page_height,
                regions=region_results,
                total_regions=len(region_results),
                ocr_regions=real_ocr_count,
                skipped_regions=empty_ocr_count,
                average_confidence=avg_confidence
            )
            
            # Save result
            self._save_page_result(document_id, page_result)
            
            logger.info(f"✅ Page {page_number} (REAL EasyOCR): {real_ocr_count} real text, {empty_ocr_count} empty")
            return page_result
            
        except Exception as e:
            logger.error(f"❌ Failed to process page {page_data.get('page_number', '?')}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_page_result(self, document_id: uuid.UUID, page_result: PageOCRResult):
        """Save REAL OCR results."""
        try:
            ocr_dir = settings.DOCUMENTS_DIR / str(document_id) / "ocr"
            ocr_dir.mkdir(parents=True, exist_ok=True)
            
            page_file = ocr_dir / f"page_{page_result.page_number}_ocr.json"
            with open(page_file, "w", encoding="utf-8") as f:
                json_str = page_result.model_dump_json(indent=2)
                f.write(json_str)
            
            logger.debug(f"Saved REAL OCR results to {page_file}")
        except Exception as e:
            logger.error(f"Failed to save OCR results: {e}")
    
    def get_status(self, document_id: uuid.UUID) -> Dict[str, Any]:
        """Get REAL OCR processing status."""
        ocr_dir = settings.DOCUMENTS_DIR / str(document_id) / "ocr"
        
        if not ocr_dir.exists():
            return {
                "status": "not_started",
                "document_id": str(document_id),
                "engine": "easyocr_real",  # CHANGED from paddleocr_real
                "real_ocr": True,
                "note": "Waiting for REAL OCR processing"
            }
        
        ocr_files = list(ocr_dir.glob("page_*_ocr.json"))
        
        if ocr_files:
            # Check if results contain real text
            real_pages = 0
            for ocr_file in ocr_files:
                try:
                    with open(ocr_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("ocr_regions", 0) > 0:
                            real_pages += 1
                except:
                    pass
            
            return {
                "status": "completed",
                "document_id": str(document_id),
                "pages_processed": len(ocr_files),
                "pages_with_real_text": real_pages,
                "engine": "easyocr_real",  # CHANGED from paddleocr_real
                "real_ocr": True
            }
        
        return {
            "status": "processing",
            "document_id": str(document_id),
            "engine": "easyocr_real",  # CHANGED from paddleocr_real
            "real_ocr": True
        }
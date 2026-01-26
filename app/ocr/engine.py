# app/ocr/engine.py - REAL EasyOCR ONLY
"""
OCR Engine - REAL EasyOCR ONLY, NO MOCK DATA.
"""
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OCRManager:
    """REAL OCR manager - EasyOCR ONLY."""
    
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self.engine_name = "EasyOCR"
        self.mode = "real"
        
        # Initialize REAL EasyOCR
        self.easy_reader = self._init_easyocr()
        
        logger.info(f"✅ REAL EasyOCR initialized (lang={lang}, GPU={use_gpu})")
    
    def _init_easyocr(self):
        """Initialize REAL EasyOCR."""
        try:
            import warnings
            warnings.filterwarnings("ignore")
            
            import easyocr
            
            logger.info("🔄 Initializing REAL EasyOCR...")
            
            reader = easyocr.Reader(
                [self.lang], 
                gpu=self.use_gpu,
                verbose=False
            )
            
            logger.info("✅ EasyOCR initialization successful")
            return reader
                
        except Exception as e:
            logger.error(f"❌ REAL EasyOCR FAILED: {e}")
            raise RuntimeError(f"REAL OCR initialization failed: {str(e)}")
    
    def extract_text(self, image: np.ndarray, region_id: str = "unknown") -> Tuple[str, float, List[Dict]]:
        """Extract REAL text from image."""
        if image is None or image.size == 0:
            return "", 0.0, []
        
        try:
            # Ensure image is RGB
            if len(image.shape) == 2:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run REAL OCR
            results = self.easy_reader.readtext(image)
            
            texts = []
            confidences = []
            word_boxes = []
            
            for (bbox, text, confidence) in results:
                if text and text.strip():
                    clean_text = text.strip()
                    texts.append(clean_text)
                    confidences.append(float(confidence))
                    word_boxes.append({
                        "text": clean_text,
                        "confidence": float(confidence),
                        "bbox": [[float(p[0]), float(p[1])] for p in bbox]
                    })
            
            if texts:
                full_text = " ".join(texts)
                avg_confidence = float(np.mean(confidences)) if confidences else 0.0
                logger.debug(f"Extracted {len(texts)} words from {region_id}")
                return full_text, avg_confidence, word_boxes
            else:
                return "", 0.0, []
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", 0.0, []
    
    def process_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process regions with REAL OCR."""
        processed = []
        
        for i, region in enumerate(regions):
            region_id = region.get("region_id", f"region_{i}")
            region_type = region.get("detection", {}).get("type", "text_block")
            image = region.get("region_image")
            
            if region_type in ["figure", "signature"]:
                region.update({
                    "ocr_text": "",
                    "ocr_confidence": 0.0,
                    "word_boxes": [],
                    "ocr_processed": False,
                    "ocr_skipped": True,
                    "skip_reason": f"Skipped {region_type}"
                })
                processed.append(region)
                continue
            
            if image is not None:
                text, confidence, word_boxes = self.extract_text(image, region_id)
                region.update({
                    "ocr_text": text,
                    "ocr_confidence": confidence,
                    "word_boxes": word_boxes,
                    "ocr_processed": len(text.strip()) > 0,
                    "error": None,
                    "ocr_skipped": False
                })
            else:
                region.update({
                    "ocr_text": "",
                    "ocr_confidence": 0.0,
                    "word_boxes": [],
                    "ocr_processed": False,
                    "error": "No image",
                    "ocr_skipped": False
                })
            
            processed.append(region)
        
        return processed
    
    def get_info(self) -> Dict[str, Any]:
        """Get REAL OCR engine info."""
        return {
            "engine": "EasyOCR",
            "mode": "real",
            "language": self.lang,
            "available": True,
            "using_real_ocr": True,
            "gpu_enabled": self.use_gpu,
            "note": "REAL EasyOCR - No mock data"
        }
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
        
        # Initialize REAL EasyOCR with timeout
        self.easy_reader = self._init_easyocr_with_timeout()
        
        logger.info(f"✅ REAL EasyOCR initialized (lang={lang}, GPU={use_gpu})")
    
    def _init_easyocr_with_timeout(self):
        """Initialize REAL EasyOCR with timeout to prevent hanging."""
        import threading
        import queue
        import warnings
        warnings.filterwarnings("ignore")
        
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def init_reader():
            try:
                import easyocr
                logger.info(f"🔄 Initializing EasyOCR (GPU={self.use_gpu}, lang={self.lang})...")
                
                # Force CPU if GPU fails
                actual_use_gpu = self.use_gpu
                try:
                    reader = easyocr.Reader(
                        [self.lang], 
                        gpu=actual_use_gpu,
                        verbose=False,
                        download_enabled=True,
                        model_storage_directory='./easyocr_models'
                    )
                    result_queue.put(reader)
                    logger.info(f"✅ EasyOCR initialized with GPU={actual_use_gpu}")
                except Exception as gpu_error:
                    logger.warning(f"GPU initialization failed, trying CPU: {gpu_error}")
                    # Try with CPU
                    reader = easyocr.Reader(
                        [self.lang], 
                        gpu=False,
                        verbose=False,
                        download_enabled=True,
                        model_storage_directory='./easyocr_models'
                    )
                    result_queue.put(reader)
                    logger.info("✅ EasyOCR initialized with CPU fallback")
                    
            except Exception as e:
                logger.error(f"EasyOCR initialization error: {e}")
                error_queue.put(e)
        
        # Start initialization in a thread
        init_thread = threading.Thread(target=init_reader)
        init_thread.daemon = True
        init_thread.start()
        
        # Wait for initialization with timeout
        init_thread.join(timeout=60)  # 60 second timeout
        
        if init_thread.is_alive():
            logger.error("❌ EasyOCR initialization timeout (60 seconds)")
            raise RuntimeError("EasyOCR initialization timeout. Check internet connection and model downloads.")
        
        if not error_queue.empty():
            error = error_queue.get()
            logger.error(f"❌ EasyOCR initialization failed: {error}")
            raise RuntimeError(f"EasyOCR initialization failed: {str(error)}")
        
        if result_queue.empty():
            logger.error("❌ EasyOCR initialization produced no result")
            raise RuntimeError("EasyOCR initialization produced no result")
        
        return result_queue.get()
    
    def extract_text(self, image: np.ndarray, region_id: str = "unknown") -> Tuple[str, float, List[Dict]]:
        """Extract REAL text from image."""
        if image is None or image.size == 0:
            logger.warning(f"Empty image for region {region_id}")
            return "", 0.0, []
        
        try:
            # Ensure image is RGB
            if len(image.shape) == 2:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to RGB
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Run REAL OCR
            results = self.easy_reader.readtext(image, detail=1)
            
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
                logger.debug(f"Extracted {len(texts)} words from {region_id}: '{full_text[:50]}...'")
                return full_text, avg_confidence, word_boxes
            else:
                logger.debug(f"No text found in region {region_id}")
                return "", 0.0, []
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {region_id}: {e}")
            return "", 0.0, []
    
    def process_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process regions with REAL OCR."""
        processed = []
        
        logger.info(f"Processing {len(regions)} regions with OCR...")
        
        for i, region in enumerate(regions):
            region_id = region.get("region_id", f"region_{i+1}")
            region_type = region.get("detection", {}).get("type", "text_block")
            image = region.get("region_image")
            
            logger.debug(f"Processing region {region_id} ({region_type})")
            
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
            
            if image is not None and image.size > 0:
                text, confidence, word_boxes = self.extract_text(image, region_id)
                region.update({
                    "ocr_text": text,
                    "ocr_confidence": confidence,
                    "word_boxes": word_boxes,
                    "ocr_processed": len(text.strip()) > 0,
                    "error": None,
                    "ocr_skipped": False
                })
                
                if text:
                    logger.debug(f"Region {region_id}: Extracted {len(text)} chars, confidence {confidence:.2f}")
                else:
                    logger.debug(f"Region {region_id}: No text extracted")
            else:
                logger.warning(f"Region {region_id}: No image available")
                region.update({
                    "ocr_text": "",
                    "ocr_confidence": 0.0,
                    "word_boxes": [],
                    "ocr_processed": False,
                    "error": "No image",
                    "ocr_skipped": False
                })
            
            processed.append(region)
        
        # Count statistics
        text_regions = sum(1 for r in processed if r.get("ocr_processed"))
        skipped_regions = sum(1 for r in processed if r.get("ocr_skipped"))
        logger.info(f"OCR processed: {text_regions} with text, {skipped_regions} skipped, {len(processed) - text_regions - skipped_regions} empty")
        
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
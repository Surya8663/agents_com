# app/vision/enhanced_detector.py
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from app.vision.detector import LayoutDetector
from app.vision.fallback_detector import FallbackLayoutDetector
from app.vision.schema import Detection

logger = logging.getLogger(__name__)


class EnhancedLayoutDetector:
    """
    Enhanced detector that combines YOLO with traditional CV fallback.
    """
    
    def __init__(self, model_loader):
        """Initialize with both detectors."""
        self.yolo_detector = LayoutDetector(model_loader)
        self.fallback_detector = FallbackLayoutDetector()
    
    def detect_page_layout(self, image_path: Path) -> Tuple[List[Detection], Tuple[int, int]]:
        """
        Detect layout elements using YOLO first, fallback if needed.
        
        Args:
            image_path: Path to the page image
            
        Returns:
            Tuple of (list of detections, (image_width, image_height))
        """
        # First try YOLO detection
        yolo_detections, size = self.yolo_detector.detect_page_layout(image_path)
        
        # If YOLO found detections, use them
        if len(yolo_detections) > 0:
            logger.info(f"YOLO found {len(yolo_detections)} detections")
            return yolo_detections, size
        
        # If YOLO failed, use fallback detector
        logger.info(f"YOLO found no detections, using fallback detector")
        fallback_detections, size = self.fallback_detector.detect_document_regions(image_path)
        
        # If fallback also failed, create at least one detection for the whole page
        if len(fallback_detections) == 0 and size != (0, 0):
            logger.info("Creating fallback detection for entire page")
            fallback_detections = self._create_whole_page_detection(size)
        
        return fallback_detections, size
    
    def _create_whole_page_detection(self, size: Tuple[int, int]) -> List[Detection]:
        """Create a detection for the entire page as fallback."""
        width, height = size
        
        from app.vision.schema import BoundingBox, Detection
        
        # Create a bounding box for the whole page
        bbox = BoundingBox(x1=0.05, y1=0.05, x2=0.95, y2=0.95)
        
        detection = Detection(
            type="text_block",
            bbox=bbox,
            confidence=0.3,  # Low confidence since it's a fallback
            page_width=width,
            page_height=height
        )
        
        return [detection]
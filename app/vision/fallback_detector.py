# app/vision/fallback_detector.py
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from skimage import feature, filters, morphology
from scipy import ndimage

from app.vision.schema import Detection, BoundingBox

logger = logging.getLogger(__name__)


class FallbackLayoutDetector:
    """
    Fallback detector using traditional computer vision techniques
    when YOLO fails to detect document elements.
    """
    
    def detect_document_regions(self, image_path: Path) -> Tuple[List[Detection], Tuple[int, int]]:
        """
        Detect document regions using traditional CV techniques.
        
        Args:
            image_path: Path to the page image
            
        Returns:
            Tuple of (list of detections, (image_width, image_height))
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return [], (0, 0)
            
            height, width = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            detections = []
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                min_area = width * height * 0.001  # 0.1% of image area
                max_area = width * height * 0.8    # 80% of image area
                area = w * h
                
                if area < min_area or area > max_area:
                    continue
                
                # Calculate aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # Determine element type based on shape and size
                element_type = self._classify_region(w, h, aspect_ratio, area, width * height)
                
                # Create normalized bounding box
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                bbox = BoundingBox.from_pixel_coords(x1, y1, x2, y2, width, height)
                
                # Calculate confidence based on region properties
                confidence = self._calculate_confidence(w, h, aspect_ratio, element_type)
                
                # Create detection
                detection = Detection(
                    type=element_type,
                    bbox=bbox,
                    confidence=confidence,
                    page_width=width,
                    page_height=height
                )
                
                detections.append(detection)
            
            logger.info(f"Fallback detector found {len(detections)} regions in {image_path.name}")
            return detections, (width, height)
            
        except Exception as e:
            logger.error(f"Error in fallback detection: {e}")
            return [], (0, 0)
    
    def _classify_region(self, width: int, height: int, aspect_ratio: float, 
                        area: float, image_area: float) -> str:
        """Classify region based on geometric properties."""
        
        # Very wide regions are likely text blocks
        if aspect_ratio > 3.0:
            return "text_block"
        
        # Very tall regions are also likely text blocks
        if aspect_ratio < 0.33:
            return "text_block"
        
        # Square-ish regions with many internal contours might be tables
        if 0.7 < aspect_ratio < 1.5 and area > image_area * 0.05:
            return "table"
        
        # Medium square regions might be figures
        if 0.8 < aspect_ratio < 1.2 and area > image_area * 0.02:
            return "figure"
        
        # Small regions might be signatures
        if area < image_area * 0.005:
            return "signature"
        
        # Default to text block
        return "text_block"
    
    def _calculate_confidence(self, width: int, height: int, aspect_ratio: float, 
                             element_type: str) -> float:
        """Calculate confidence score based on region properties."""
        
        # Base confidence
        confidence = 0.5
        
        # Adjust based on element type
        if element_type == "text_block":
            # Text blocks should be elongated
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                confidence += 0.3
        elif element_type == "table":
            # Tables should be somewhat square
            if 0.8 < aspect_ratio < 1.2:
                confidence += 0.3
        elif element_type == "figure":
            # Figures should be roughly square
            if 0.7 < aspect_ratio < 1.3:
                confidence += 0.3
        elif element_type == "signature":
            # Signatures are small
            if width * height < 5000:  # Less than 5000 pixels
                confidence += 0.3
        
        return min(confidence, 0.95)  # Cap at 0.95
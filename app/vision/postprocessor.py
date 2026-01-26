# app/vision/postprocessor.py
import logging
from typing import List, Dict
import numpy as np

from app.vision.schema import Detection, BoundingBox

logger = logging.getLogger(__name__)


class LayoutPostProcessor:
    """Post-processes layout detections to clean and refine results."""
    
    def __init__(self, iou_threshold: float = 0.5, min_area: float = 0.001):
        """
        Initialize post-processor.
        
        Args:
            iou_threshold: IoU threshold for non-maximum suppression
            min_area: Minimum normalized area for valid detections (0-1)
        """
        self.iou_threshold = iou_threshold
        self.min_area = min_area
    
    def filter_by_confidence(self, detections: List[Detection], 
                           min_confidence: Dict[str, float] = None) -> List[Detection]:
        """
        Filter detections by confidence thresholds.
        
        Args:
            detections: List of detections
            min_confidence: Dict mapping element types to minimum confidence
            
        Returns:
            Filtered list of detections
        """
        if min_confidence is None:
            min_confidence = {
                "text_block": 0.25,
                "table": 0.3,
                "figure": 0.35,
                "signature": 0.4
            }
        
        filtered = []
        for detection in detections:
            threshold = min_confidence.get(detection.type, 0.25)
            if detection.confidence >= threshold:
                filtered.append(detection)
        
        logger.debug(f"Filtered from {len(detections)} to {len(filtered)} detections by confidence")
        return filtered
    
    def filter_by_area(self, detections: List[Detection]) -> List[Detection]:
        """
        Remove detections that are too small.
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for detection in detections:
            if detection.bbox.area() >= self.min_area:
                filtered.append(detection)
        
        logger.debug(f"Filtered from {len(detections)} to {len(filtered)} detections by area")
        return filtered
    
    def non_maximum_suppression(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply non-maximum suppression to remove overlapping boxes.
        
        Args:
            detections: List of detections
            
        Returns:
            List of detections after NMS
        """
        if not detections:
            return []
        
        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        keep = []
        
        while sorted_detections:
            # Take the detection with highest confidence
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # Find and remove overlapping detections
            to_remove = []
            for i, other in enumerate(sorted_detections):
                if current.bbox.iou(other.bbox) > self.iou_threshold:
                    to_remove.append(i)
            
            # Remove from highest index to lowest to maintain indices
            for i in reversed(to_remove):
                sorted_detections.pop(i)
        
        logger.debug(f"NMS filtered from {len(detections)} to {len(keep)} detections")
        return keep
    
    def merge_text_blocks(self, detections: List[Detection], 
                         vertical_threshold: float = 0.05,
                         horizontal_threshold: float = 0.1) -> List[Detection]:
        """
        Merge closely spaced text blocks into larger text regions.
        
        Args:
            detections: List of detections
            vertical_threshold: Maximum vertical gap for merging (normalized)
            horizontal_threshold: Maximum horizontal gap for merging (normalized)
            
        Returns:
            List of detections with merged text blocks
        """
        text_blocks = [d for d in detections if d.type == "text_block"]
        other_detections = [d for d in detections if d.type != "text_block"]
        
        if not text_blocks:
            return detections
        
        # Sort text blocks by y1 coordinate (top to bottom)
        text_blocks.sort(key=lambda x: x.bbox.y1)
        
        merged = []
        current_group = [text_blocks[0]]
        
        for block in text_blocks[1:]:
            last_block = current_group[-1]
            
            # Check if blocks are in same horizontal band
            vertical_overlap = (block.bbox.y1 <= last_block.bbox.y2 + vertical_threshold)
            
            # Check if blocks are close horizontally
            horizontal_gap = max(0, block.bbox.x1 - last_block.bbox.x2)
            horizontal_close = horizontal_gap <= horizontal_threshold
            
            if vertical_overlap and horizontal_close:
                current_group.append(block)
            else:
                # Merge current group
                if len(current_group) > 1:
                    merged_block = self._merge_detection_group(current_group)
                    merged.append(merged_block)
                else:
                    merged.extend(current_group)
                
                # Start new group
                current_group = [block]
        
        # Merge last group
        if len(current_group) > 1:
            merged_block = self._merge_detection_group(current_group)
            merged.append(merged_block)
        else:
            merged.extend(current_group)
        
        logger.debug(f"Merged {len(text_blocks)} text blocks into {len(merged)} regions")
        
        return merged + other_detections
    
    def _merge_detection_group(self, group: List[Detection]) -> Detection:
        """Merge a group of detections into a single detection."""
        # Use the first detection as template
        template = group[0]
        
        # Calculate merged bounding box
        x1 = min(d.bbox.x1 for d in group)
        y1 = min(d.bbox.y1 for d in group)
        x2 = max(d.bbox.x2 for d in group)
        y2 = max(d.bbox.y2 for d in group)
        
        # Average confidence
        avg_confidence = sum(d.confidence for d in group) / len(group)
        
        # Create merged detection
        merged_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        
        return Detection(
            type=template.type,
            bbox=merged_bbox,
            confidence=avg_confidence,
            page_width=template.page_width,
            page_height=template.page_height
        )
    
    def process(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply complete post-processing pipeline.
        
        Args:
            detections: Raw detections from model
            
        Returns:
            Cleaned and refined detections
        """
        if not detections:
            return []
        
        # Apply processing steps
        detections = self.filter_by_area(detections)
        detections = self.filter_by_confidence(detections)
        detections = self.merge_text_blocks(detections)
        detections = self.non_maximum_suppression(detections)
        
        return detections
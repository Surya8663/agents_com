# app/ocr/region_cropper.py (UPDATED)
"""
Region cropping utilities for OCR.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from app.config.settings import settings
# REMOVE: from app.vision.schema import Detection  # This imports PyTorch

logger = logging.getLogger(__name__)


class RegionCropper:
    """Handles cropping of regions from page images."""
    
    def __init__(self, padding: int = 5):
        self.padding = padding
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image using PIL (works with or without OpenCV)."""
        try:
            # Convert to Path object if string
            if isinstance(image_path, str):
                image_path = Path(image_path)
        
        # Try multiple path resolutions
            possible_paths = [
                image_path,  # Original path
                Path("data") / image_path,  # Add data prefix
                settings.BASE_DATA_DIR / image_path,  # Use BASE_DATA_DIR
            ]
        
            actual_path = None
            for path in possible_paths:
                if path.exists():
                    actual_path = path
                    break
        
            if actual_path is None:
                # Try to find relative to project root
                project_root = Path(__file__).parent.parent.parent
                project_path = project_root / image_path
                if project_path.exists():
                    actual_path = project_path
                else:
                    logger.error(f"Image not found. Tried: {image_path}")
                    return None
        
            # Load with PIL
            from PIL import Image
            img = Image.open(actual_path)
        
        # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
            # Convert to numpy array
            img_array = np.array(img)
            logger.debug(f"Loaded image: {actual_path.name}, shape: {img_array.shape}")
            return img_array
        
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def crop_region(self, image: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """Crop a region from image using detection dict (not Detection object)."""
        if image is None:
            logger.error("Cannot crop: image is None")
            return None
        
        try:
            height, width = image.shape[:2]
            
            # Extract bbox from detection dict
            bbox = detection.get("bbox", {})
            x1 = float(bbox.get("x1", 0))
            y1 = float(bbox.get("y1", 0))
            x2 = float(bbox.get("x2", 0))
            y2 = float(bbox.get("y2", 0))
            
            # Convert normalized coordinates to pixels
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            
            # Add padding
            x1_px = max(0, x1_px - self.padding)
            y1_px = max(0, y1_px - self.padding)
            x2_px = min(width - 1, x2_px + self.padding)
            y2_px = min(height - 1, y2_px + self.padding)
            
            # Ensure valid region
            if x2_px <= x1_px or y2_px <= y1_px:
                logger.warning(f"Invalid region: ({x1_px},{y1_px}) to ({x2_px},{y2_px}) for image {width}x{height}")
                return None
            
            # Crop the region
            cropped = image[y1_px:y2_px, x1_px:x2_px]
            
            if cropped.size == 0:
                logger.warning(f"Empty crop: {x1_px},{y1_px} to {x2_px},{y2_px}")
                return None
            
            logger.debug(f"Cropped region {detection.get('type', 'unknown')}: {cropped.shape}")
            return cropped
            
        except Exception as e:
            logger.error(f"Failed to crop region: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def crop_regions_from_page(self, image_path: Path, detections: List[dict]) -> List[Dict[str, Any]]:
        """Crop all regions from a page using detection dicts."""
        full_image = self.load_image(image_path)
        if full_image is None:
            logger.error(f"Failed to load {image_path}")
            return []
        
        regions = []
        
        for i, detection in enumerate(detections):
            region_id = f"r{i+1}"
            region_image = self.crop_region(full_image, detection)
            
            # Determine region type
            region_type = detection.get("type", "text_block")
            
            # Determine if we should process this region
            should_process = region_type in ["text_block", "table"]
            skip_reason = "" if should_process else f"Skipping {region_type}"
            
            regions.append({
                "region_id": region_id,
                "detection": detection,  # Keep as dict
                "region_image": region_image,
                "should_process": should_process,
                "skip_reason": skip_reason,
                "image_available": region_image is not None
            })
        
        # Log statistics
        text_blocks = [r for r in regions if r["detection"].get("type") == "text_block"]
        tables = [r for r in regions if r["detection"].get("type") == "table"]
        figures = [r for r in regions if r["detection"].get("type") == "figure"]
        signatures = [r for r in regions if r["detection"].get("type") == "signature"]
        
        logger.info(f"Cropped {len(regions)} regions from {image_path.name}: "
                   f"{len(text_blocks)} text blocks, {len(tables)} tables, "
                   f"{len(figures)} figures, {len(signatures)} signatures")
        
        return regions
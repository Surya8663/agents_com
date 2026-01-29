# app/vision/detector.py - CORRECT VERSION (no changes needed)
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

from app.vision.schema import Detection, BoundingBox
from app.vision.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Performs layout detection on document page images."""
    
    # Lower confidence thresholds for document detection
    CONFIDENCE_THRESHOLDS = {
        "text_block": 0.15,    # Lower threshold for text
        "table": 0.20,         # Slightly higher for tables
        "figure": 0.25,        # Higher for figures
        "signature": 0.30      # Highest for signatures
    }
    
    # Image preprocessing settings
    PREPROCESS_ENABLED = True
    CONTRAST_ALPHA = 1.5       # Contrast control (1.0-3.0)
    CONTRAST_BETA = 0          # Brightness control
    GRAYSCALE_THRESHOLD = True  # Convert to grayscale and threshold
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize layout detector.
        
        Args:
            model_loader: Pre-loaded YOLOv8 model loader
        """
        self.model_loader = model_loader
        self.model = model_loader.model
        
    def preprocess_image(self, image_cv: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve detection on documents.
        
        Args:
            image_cv: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image
        """
        if not self.PREPROCESS_ENABLED:
            return image_cv
        
        try:
            # Convert to grayscale for document images
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold to get binary image
            # This helps with scanned documents
            binary = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to 3 channels
            processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            return processed
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_cv
    
    def detect_page_layout(self, image_path: Path) -> Tuple[List[Detection], Tuple[int, int]]:
        """
        Detect layout elements in a single page image.
        
        Args:
            image_path: Path to the page image
            
        Returns:
            Tuple of (list of detections, (image_width, image_height))
        """
        if not self.model_loader.is_loaded():
            logger.error("YOLO model not loaded")
            return [], (0, 0)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_width, image_height = image.size
            
            # Convert PIL to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess for better document detection
            processed_image = self.preprocess_image(image_cv)
            
            # Run YOLO inference with adjusted settings for documents
            results = self.model(
                processed_image, 
                conf=0.15,  # Lower confidence for documents
                iou=0.4,    # Lower IoU for overlapping text
                verbose=False,
                augment=True  # Use augmentation for better detection
            )
            
            detections = []
            
            for result in results:
                if result.boxes is None:
                    logger.debug(f"No detections found in {image_path.name}")
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                
                # Get class names if available
                class_names = []
                if hasattr(self.model, 'names'):
                    for class_id in class_ids:
                        class_names.append(self.model.names.get(int(class_id), ""))
                
                for idx, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    # Convert numpy types to Python native types
                    x1, y1, x2, y2 = map(float, box)
                    confidence = float(confidence)
                    
                    # Get class name for better mapping
                    class_name = class_names[idx] if idx < len(class_names) else ""
                    
                    # Map YOLO class to document element type
                    element_type = self.model_loader.map_class_id_to_type(
                        int(class_id), 
                        class_name
                    )
                    
                    # Check confidence threshold for this element type
                    threshold = self.CONFIDENCE_THRESHOLDS.get(element_type, 0.15)
                    if confidence < threshold:
                        continue
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, image_width - 1))
                    y1 = max(0, min(y1, image_height - 1))
                    x2 = max(0, min(x2, image_width - 1))
                    y2 = max(0, min(y2, image_height - 1))
                    
                    # Skip if box is too small
                    box_width = x2 - x1
                    box_height = y2 - y1
                    min_box_size = min(image_width, image_height) * 0.01  # 1% of smallest dimension
                    
                    if box_width < min_box_size or box_height < min_box_size:
                        continue
                    
                    # Create normalized bounding box
                    bbox = BoundingBox.from_pixel_coords(
                        int(x1), int(y1), int(x2), int(y2),
                        image_width, image_height
                    )
                    
                    # Skip if normalized area is too small
                    if bbox.area() < 0.001:  # Less than 0.1% of image
                        continue
                    
                    # Create detection
                    detection = Detection(
                        type=element_type,
                        bbox=bbox,
                        confidence=confidence,
                        page_width=image_width,
                        page_height=image_height
                    )
                    
                    detections.append(detection)
            
            logger.info(f"Found {len(detections)} layout elements in {image_path.name}")
            return detections, (image_width, image_height)
            
        except Exception as e:
            logger.error(f"Error detecting layout in {image_path}: {e}")
            return [], (0, 0)
    
    def detect_with_grid(self, image_path: Path, grid_size: int = 5) -> Tuple[List[Detection], Tuple[int, int]]:
        """
        Alternative detection method: Divide image into grid and detect each cell.
        This can help when full-page detection fails.
        """
        if not self.model_loader.is_loaded():
            logger.error("YOLO model not loaded")
            return [], (0, 0)
        
        try:
            # Load image
            image = Image.open(image_path)
            image_width, image_height = image.size
            
            # Convert to OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Calculate grid cell size
            cell_width = image_width // grid_size
            cell_height = image_height // grid_size
            
            all_detections = []
            
            # Process each grid cell
            for row in range(grid_size):
                for col in range(grid_size):
                    # Calculate cell coordinates
                    x1 = col * cell_width
                    y1 = row * cell_height
                    x2 = min((col + 1) * cell_width, image_width)
                    y2 = min((row + 1) * cell_height, image_height)
                    
                    # Extract cell
                    cell = image_cv[y1:y2, x1:x2]
                    
                    if cell.size == 0:
                        continue
                    
                    # Run detection on cell
                    results = self.model(cell, conf=0.15, verbose=False)
                    
                    for result in results:
                        if result.boxes is None:
                            continue
                        
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, confidence, class_id in zip(boxes, confidences, class_ids):
                            # Adjust coordinates to full image
                            box_x1, box_y1, box_x2, box_y2 = map(float, box)
                            
                            # Convert to full image coordinates
                            full_x1 = x1 + box_x1
                            full_y1 = y1 + box_y1
                            full_x2 = x1 + box_x2
                            full_y2 = y1 + box_y2
                            
                            # Get element type
                            element_type = self.model_loader.map_class_id_to_type(int(class_id))
                            
                            # Create bounding box
                            bbox = BoundingBox.from_pixel_coords(
                                int(full_x1), int(full_y1), int(full_x2), int(full_y2),
                                image_width, image_height
                            )
                            
                            # Create detection
                            detection = Detection(
                                type=element_type,
                                bbox=bbox,
                                confidence=float(confidence),
                                page_width=image_width,
                                page_height=image_height
                            )
                            
                            all_detections.append(detection)
            
            logger.info(f"Grid detection found {len(all_detections)} elements in {image_path.name}")
            return all_detections, (image_width, image_height)
            
        except Exception as e:
            logger.error(f"Error in grid detection for {image_path}: {e}")
            return [], (0, 0)
    
    def batch_detect(self, image_paths: List[Path]) -> List[Tuple[List[Detection], Tuple[int, int]]]:
        """
        Detect layout elements in multiple images.
        
        Args:
            image_paths: List of paths to page images
            
        Returns:
            List of (detections, (width, height)) for each image
        """
        results = []
        for img_path in image_paths:
            # Try regular detection first
            detections, size = self.detect_page_layout(img_path)
            
            # If no detections, try grid-based detection
            if len(detections) == 0:
                logger.info(f"No detections with regular method, trying grid method for {img_path.name}")
                detections, size = self.detect_with_grid(img_path)
            
            results.append((detections, size))
        return results
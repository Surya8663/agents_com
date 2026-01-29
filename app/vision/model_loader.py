# app/vision/model_loader.py - REAL MODEL ONLY - FIXED VERSION
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of YOLOv8 models - REAL ONLY."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLOv8 model loader.
        
        Args:
            model_path: Path to YOLOv8 .pt model file
        """
        from app.config.settings import settings
        
        # Use provided path or default from settings
        self.model_path = model_path or settings.LAYOUT_MODEL_PATH
        self.model = None
        self.loaded = False
        self.error = None
        
        logger.info(f"ModelLoader initialized for: {self.model_path}")
    
    def load(self) -> bool:
        """
        Load REAL YOLOv8 model.
        
        Returns:
            bool: True if model loaded successfully, False if real model fails
        """
        try:
            # Check if model file exists
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                self.error = f"Model file not found: {self.model_path}"
                return False
            
            logger.info(f"Loading REAL YOLOv8 model from: {self.model_path}")
            
            # Import YOLO
            from ultralytics import YOLO
            
            # Load model WITHOUT training or data parameter
            self.model = YOLO(str(model_file))
            
            # Set to evaluation mode
            self.model.eval()
            self.loaded = True
            
            logger.info(f"✅ REAL YOLOv8 model loaded successfully")
            logger.info(f"   Model info: {self.get_model_info()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load REAL YOLO model: {e}")
            import traceback
            traceback.print_exc()
            
            self.error = str(e)
            self.loaded = False
            return False  # Return False - REAL model failed
    
    def get_model(self):
        """Get the REAL model (loads if not already loaded)."""
        if not self.loaded:
            success = self.load()
            if not success:
                raise RuntimeError(f"Failed to load REAL model: {self.error}")
        return self.model
    
    def get_model_info(self) -> dict:
        """Get REAL model information."""
        info = {
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "has_real_model": self.loaded,
            "available": self.loaded,
            "error": self.error
        }
        
        if self.loaded and self.model:
            try:
                info["model_type"] = type(self.model).__name__
                if hasattr(self.model, 'names') and self.model.names:
                    info["num_classes"] = len(self.model.names)
                    info["classes"] = {i: name for i, name in enumerate(self.model.names.values())}
                else:
                    info["num_classes"] = "unknown"
                    info["classes"] = {}
            except:
                info["model_type"] = "YOLO"
                info["num_classes"] = "unknown"
        
        return info
    
    def is_loaded(self) -> bool:
        """Check if REAL model is loaded."""
        return self.loaded and self.model is not None
    
    def map_class_id_to_type(self, class_id: int, class_name: str = "") -> str:
        """
        Map YOLO COCO class IDs to document element types.
        
        YOLOv8n is trained on COCO (80 classes of objects).
        We need to map these to document layout elements.
        
        Args:
            class_id: YOLO class ID (0-79 for COCO)
            class_name: Optional class name from model
            
        Returns:
            Document element type
        """
        # YOLOv8 COCO classes mapping to document elements
        # This is a simple mapping - for real document layout detection,
        # you need a model trained on document layout data
        
        if class_name:
            class_name_lower = class_name.lower()
            
            # Map based on class names
            text_indicators = ['book', 'laptop', 'cell phone', 'tv', 'remote']
            table_indicators = ['dining table', 'bench']
            figure_indicators = ['picture', 'vase', 'clock']
            signature_indicators = ['person']
            
            for indicator in text_indicators:
                if indicator in class_name_lower:
                    return "text_block"
            
            for indicator in table_indicators:
                if indicator in class_name_lower:
                    return "table"
            
            for indicator in figure_indicators:
                if indicator in class_name_lower:
                    return "figure"
            
            for indicator in signature_indicators:
                if indicator in class_name_lower:
                    return "signature"
        
        # Default mapping by class ID ranges
        if 0 <= class_id <= 20:    # Person, vehicle, animal categories
            return "signature" if class_id == 0 else "figure"
        elif 21 <= class_id <= 40:  # Food, furniture
            return "table" if class_id in [60, 61] else "figure"  # dining table, bed
        elif 41 <= class_id <= 60:  # Electronic, kitchen items
            return "text_block" if class_id in [63, 67, 73] else "figure"  # laptop, cell phone, book
        else:                       # Miscellaneous
            return "figure"
        
        # Default fallback
        return "text_block"
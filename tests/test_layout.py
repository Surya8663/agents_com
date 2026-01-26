# tests/test_layout.py
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import uuid

import numpy as np
from PIL import Image

from app.vision.schema import BoundingBox, Detection, PageLayout
from app.vision.model_loader import ModelLoader
from app.vision.postprocessor import LayoutPostProcessor
from app.api.layout import LayoutAnalyzer


class TestBoundingBox:
    """Test bounding box functionality."""
    
    def test_from_pixel_coords(self):
        """Test conversion from pixel to normalized coordinates."""
        bbox = BoundingBox.from_pixel_coords(100, 150, 300, 350, 1000, 500)
        
        assert bbox.x1 == 0.1  # 100/1000
        assert bbox.y1 == 0.3  # 150/500
        assert bbox.x2 == 0.3  # 300/1000
        assert bbox.y2 == 0.7  # 350/500
    
    def test_to_pixel_coords(self):
        """Test conversion from normalized to pixel coordinates."""
        bbox = BoundingBox(x1=0.1, y1=0.3, x2=0.3, y2=0.7)
        pixel_coords = bbox.to_pixel_coords(1000, 500)
        
        assert pixel_coords == (100, 150, 300, 350)  # 0.1*1000=100, etc.
    
    def test_area(self):
        """Test area calculation."""
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.4, y2=0.6)
        area = bbox.area()
        
        expected_area = (0.4 - 0.1) * (0.6 - 0.2)  # 0.3 * 0.4 = 0.12
        assert area == pytest.approx(expected_area)
    
    def test_iou(self):
        """Test IoU calculation."""
        bbox1 = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)  # Area: 0.4 * 0.4 = 0.16
        bbox2 = BoundingBox(x1=0.3, y1=0.3, x2=0.7, y2=0.7)  # Area: 0.4 * 0.4 = 0.16
        
        # Intersection: [0.3, 0.3, 0.5, 0.5] -> Area: 0.2 * 0.2 = 0.04
        # Union: 0.16 + 0.16 - 0.04 = 0.28
        # IoU: 0.04 / 0.28 ≈ 0.142857
        iou = bbox1.iou(bbox2)
        assert iou == pytest.approx(0.142857, rel=1e-5)


class TestLayoutPostProcessor:
    """Test post-processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = LayoutPostProcessor(iou_threshold=0.5, min_area=0.01)
        
        # Create test detections
        self.detections = [
            Detection(
                type="text_block",
                bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3),
                confidence=0.8,
                page_width=1000,
                page_height=1000
            ),
            Detection(
                type="text_block",
                bbox=BoundingBox(x1=0.15, y1=0.15, x2=0.35, y2=0.35),  # Overlaps with first
                confidence=0.7,
                page_width=1000,
                page_height=1000
            ),
            Detection(
                type="table",
                bbox=BoundingBox(x1=0.6, y1=0.6, x2=0.9, y2=0.9),
                confidence=0.9,
                page_width=1000,
                page_height=1000
            ),
        ]
    
    def test_filter_by_confidence(self):
        """Test confidence filtering."""
        filtered = self.processor.filter_by_confidence(
            self.detections,
            {"text_block": 0.75, "table": 0.85}  # Higher thresholds
        )
        
        # Only the table (0.9 > 0.85) should remain
        assert len(filtered) == 1
        assert filtered[0].type == "table"
    
    def test_filter_by_area(self):
        """Test area filtering."""
        # Create a very small detection
        small_detection = Detection(
            type="text_block",
            bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.101, y2=0.101),  # Area: 0.000001
            confidence=0.8,
            page_width=1000,
            page_height=1000
        )
        
        detections = self.detections + [small_detection]
        filtered = self.processor.filter_by_area(detections)
        
        # Small detection should be filtered out
        assert len(filtered) == len(self.detections)
    
    def test_non_maximum_suppression(self):
        """Test NMS for overlapping detections."""
        # First two detections overlap heavily
        result = self.processor.non_maximum_suppression(self.detections)
        
        # Should keep higher confidence text block (0.8) and the table
        assert len(result) == 2
        
        # Check types and confidences
        types = {d.type for d in result}
        confidences = {d.confidence for d in result}
        
        assert types == {"text_block", "table"}
        assert 0.8 in confidences  # Higher confidence text block
        assert 0.9 in confidences  # Table
    
    def test_merge_text_blocks(self):
        """Test text block merging."""
        # Create vertically aligned text blocks
        text_blocks = [
            Detection(
                type="text_block",
                bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.15),
                confidence=0.8,
                page_width=1000,
                page_height=1000
            ),
            Detection(
                type="text_block",
                bbox=BoundingBox(x1=0.1, y1=0.16, x2=0.4, y2=0.21),  # Close vertically
                confidence=0.7,
                page_width=1000,
                page_height=1000
            ),
        ]
        
        merged = self.processor.merge_text_blocks(text_blocks, vertical_threshold=0.05)
        
        # Should merge into one text block
        assert len(merged) == 1
        assert merged[0].type == "text_block"
        
        # Check merged bounding box
        bbox = merged[0].bbox
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.1  # Minimum y1
        assert bbox.x2 == 0.4
        assert bbox.y2 == 0.21  # Maximum y2


class TestLayoutAnalyzer:
    """Test the main layout analyzer."""
    
    def test_get_layout_status(self, tmp_path):
        """Test layout status checking."""
        # Mock settings
        from app.config import settings
        
        # Create test document structure
        document_id = uuid.uuid4()
        doc_dir = tmp_path / "documents" / str(document_id)
        layout_dir = doc_dir / "layout"
        
        # Create directories
        layout_dir.mkdir(parents=True)
        
        # Mock settings.DOCUMENTS_DIR
        settings.DOCUMENTS_DIR = tmp_path / "documents"
        
        analyzer = LayoutAnalyzer()
        
        # Test not started
        status = analyzer.get_layout_status(document_id)
        assert status["status"] == "not_started"
        
        # Test partial
        (layout_dir / "page_1_layout.json").touch()
        status = analyzer.get_layout_status(document_id)
        assert status["status"] == "partial"
        assert status["pages_processed"] == 1
        
        # Test completed
        (layout_dir / "document_layout_summary.json").touch()
        status = analyzer.get_layout_status(document_id)
        assert status["status"] == "completed"


def test_model_loader():
    """Test model loader initialization."""
    with patch('ultralytics.YOLO') as mock_yolo:
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        mock_model.names = {0: "text", 1: "table"}
        
        loader = ModelLoader("test_model.pt")
        loaded = loader.load()
        
        assert loaded is True
        assert loader.is_loaded() is True
        
        # Test model info
        info = loader.get_model_info()
        assert info["status"] == "loaded"
        assert "test_model.pt" in info["model_name"]
        
        # Test class mapping
        assert loader.map_class_id_to_type(0) == "text_block"
        assert loader.map_class_id_to_type(1) == "table"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
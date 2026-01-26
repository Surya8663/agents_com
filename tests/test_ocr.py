"""
Tests for Phase 3 OCR functionality.
"""
import pytest
import uuid
import json
from pathlib import Path

from app.ocr.schema import OCRBoundingBox, OCRRegionResult, PageOCRResult
from app.vision.schema import BoundingBox


class TestOCRSchema:
    """Test OCR schema/models."""
    
    def test_ocr_bounding_box_creation(self):
        """Test OCRBoundingBox creation."""
        bbox = OCRBoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.3)
        
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.1
        assert bbox.x2 == 0.4
        assert bbox.y2 == 0.3
        
        # Test serialization
        data = bbox.model_dump()
        assert data["x1"] == 0.1
        assert data["y2"] == 0.3
    
    def test_region_result_creation(self):
        """Test OCRRegionResult creation."""
        bbox = OCRBoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.3)
        
        region = OCRRegionResult(
            region_id="r1",
            type="text_block",
            bbox=bbox,
            ocr_text="Sample extracted text",
            ocr_confidence=0.95,
            engine="PaddleOCR",
            language="en",
            ocr_processed=True
        )
        
        assert region.region_id == "r1"
        assert region.type == "text_block"
        assert region.ocr_text == "Sample extracted text"
        assert region.ocr_confidence == 0.95
        assert region.engine == "PaddleOCR"
        assert region.ocr_processed is True
    
    def test_page_result_creation(self):
        """Test PageOCRResult creation."""
        # Create sample regions
        bbox1 = OCRBoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.3)
        bbox2 = OCRBoundingBox(x1=0.5, y1=0.1, x2=0.8, y2=0.3)
        
        regions = [
            OCRRegionResult(
                region_id="r1",
                type="text_block",
                bbox=bbox1,
                ocr_text="Text 1",
                ocr_confidence=0.9,
                ocr_processed=True
            ),
            OCRRegionResult(
                region_id="r2",
                type="text_block",
                bbox=bbox2,
                ocr_text="Text 2",
                ocr_confidence=0.85,
                ocr_processed=True
            )
        ]
        
        doc_id = uuid.uuid4()
        page_result = PageOCRResult(
            page_number=1,
            document_id=doc_id,
            image_path="test/path.png",
            image_width=1000,
            image_height=1500,
            regions=regions,
            total_regions=2,
            ocr_regions=2,
            skipped_regions=0,
            average_confidence=0.875
        )
        
        assert page_result.page_number == 1
        assert page_result.document_id == doc_id
        assert len(page_result.regions) == 2
        assert page_result.total_regions == 2
        assert page_result.ocr_regions == 2
        assert page_result.average_confidence == 0.875
        
        # Test serialization
        data = page_result.model_dump()
        assert data["page_number"] == 1
        assert len(data["regions"]) == 2


class TestOCRIntegration:
    """Integration tests for OCR functionality."""
    
    def test_engine_import(self):
        """Test that OCR engine imports correctly."""
        try:
            from app.ocr.engine import OCRManager
            manager = OCRManager()
            
            # Should work even with missing dependencies
            info = manager.get_engine_info()
            assert "engine" in info
            assert "engine_type" in info
            
            print(f"Engine type: {info['engine_type']}")
            print(f"Using mock: {info.get('using_mock', False)}")
            
        except ImportError as e:
            pytest.skip(f"Import error: {e}")
    
    def test_schema_serialization(self):
        """Test that schemas can be serialized to JSON."""
        # Create a complete OCR result
        bbox = OCRBoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.3)
        
        region = OCRRegionResult(
            region_id="r1",
            type="text_block",
            bbox=bbox,
            ocr_text="Test text",
            ocr_confidence=0.9,
            ocr_processed=True
        )
        
        page_result = PageOCRResult(
            page_number=1,
            document_id=uuid.uuid4(),
            image_path="test.png",
            image_width=1000,
            image_height=1500,
            regions=[region],
            total_regions=1,
            ocr_regions=1,
            skipped_regions=0,
            average_confidence=0.9
        )
        
        # Serialize to JSON
        json_str = page_result.model_dump_json(indent=2)
        data = json.loads(json_str)
        
        assert data["page_number"] == 1
        assert len(data["regions"]) == 1
        assert data["regions"][0]["ocr_text"] == "Test text"
        
        print("✅ Schema serialization test passed")


if __name__ == "__main__":
    print("Running OCR tests...")
    
    # Run tests
    tester = TestOCRSchema()
    tester.test_ocr_bounding_box_creation()
    tester.test_region_result_creation()
    tester.test_page_result_creation()
    
    tester = TestOCRIntegration()
    tester.test_engine_import()
    tester.test_schema_serialization()
    
    print("\n✅ All OCR tests completed!")
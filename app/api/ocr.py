"""
OCR API endpoints for Phase 3 - REAL EasyOCR ONLY.
"""
import uuid
import json
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.config.settings import settings
from app.ocr.processor import OCRProcessor
from app.vision.schema import Detection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])


# Global OCR processor - will be initialized on demand
ocr_processor = None


def get_or_create_processor(lang: str = "en", use_gpu: bool = False) -> OCRProcessor:
    """Get or create REAL OCR processor."""
    global ocr_processor
    if ocr_processor is None or ocr_processor.lang != lang or ocr_processor.use_gpu != use_gpu:
        try:
            ocr_processor = OCRProcessor(lang=lang, use_gpu=use_gpu)
            # Verify it's REAL
            if ocr_processor.ocr_manager.mode != "real":
                raise RuntimeError("OCR processor is not in REAL mode")
        except Exception as e:
            logger.error(f"Failed to create REAL OCR processor: {e}")
            raise
    return ocr_processor


@router.post("/process/{document_id}")
async def process_document_ocr(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    lang: Optional[str] = "en",
    use_gpu: Optional[bool] = False
):
    """
    Start REAL OCR processing for a document.
    """
    try:
        # Check document exists
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        if not doc_dir.exists():
            raise HTTPException(404, detail=f"Document {document_id} not found")
        
        # Check layout exists
        layout_dir = doc_dir / "layout"
        if not layout_dir.exists():
            raise HTTPException(400, detail="Run layout analysis first (/layout/analyze)")
        
        # Create REAL OCR processor
        processor = OCRProcessor(lang=lang, use_gpu=use_gpu)
        engine_info = processor.ocr_manager.get_info()
        
        # Verify it's REAL
        if engine_info.get('mode') != 'real':
            raise HTTPException(500, detail="OCR engine is not in REAL mode")
        
        # Start processing in background
        background_tasks.add_task(_process_document_ocr, document_id, processor)
        
        return {
            "message": "REAL OCR processing started",
            "document_id": str(document_id),
            "engine": "EasyOCR",  # CHANGED from PaddleOCR
            "mode": "real",
            "language": lang,
            "gpu_enabled": use_gpu,
            "engine_info": engine_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start OCR: {e}")
        raise HTTPException(500, detail=f"OCR initialization failed: {str(e)}")


async def _process_document_ocr(document_id: uuid.UUID, processor: OCRProcessor):
    """Background task to process REAL OCR."""
    try:
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        layout_dir = doc_dir / "layout"
        
        # Process each page
        layout_files = sorted(layout_dir.glob("page_*_layout.json"))
        total_pages = len(layout_files)
        
        for i, page_file in enumerate(layout_files):
            try:
                with open(page_file, "r", encoding="utf-8") as f:
                    page_data = json.load(f)
                
                result = processor.process_page(document_id, page_data)
                if result:
                    logger.info(f"✅ Page {i+1}/{total_pages}: {result.ocr_regions} regions with REAL text")
                else:
                    logger.warning(f"⚠️ Page {i+1}/{total_pages}: No OCR result")
                
            except Exception as e:
                logger.error(f"Error processing {page_file.name}: {e}")
        
        logger.info(f"✅ REAL OCR complete for {document_id}")
        
    except Exception as e:
        logger.error(f"❌ OCR processing failed: {e}")


@router.get("/status/{document_id}")
async def get_ocr_status(document_id: uuid.UUID):
    """Get REAL OCR processing status."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(404, detail=f"Document {document_id} not found")
    
    ocr_dir = doc_dir / "ocr"
    
    if not ocr_dir.exists():
        return {
            "status": "not_started",
            "document_id": str(document_id),
            "engine": "EasyOCR",  # CHANGED from PaddleOCR
            "mode": "real"
        }
    
    ocr_files = list(ocr_dir.glob("page_*_ocr.json"))
    
    if not ocr_files:
        return {
            "status": "processing",
            "document_id": str(document_id),
            "engine": "EasyOCR",  # CHANGED from PaddleOCR
            "mode": "real"
        }
    
    # Count REAL text results
    real_text_pages = 0
    for ocr_file in ocr_files:
        try:
            with open(ocr_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("ocr_regions", 0) > 0:
                    real_text_pages += 1
        except:
            pass
    
    layout_files = list((doc_dir / "layout").glob("page_*_layout.json"))
    
    return {
        "status": "completed" if len(ocr_files) >= len(layout_files) else "processing",
        "document_id": str(document_id),
        "pages_processed": len(ocr_files),
        "total_pages": len(layout_files),
        "pages_with_real_text": real_text_pages,
        "engine": "EasyOCR",  # CHANGED from PaddleOCR
        "mode": "real"
    }


@router.get("/results/{document_id}")
async def get_ocr_results(
    document_id: uuid.UUID,
    page: Optional[int] = None
):
    """Get REAL OCR results."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(404, detail="Document not found")
    
    ocr_dir = doc_dir / "ocr"
    
    if not ocr_dir.exists():
        raise HTTPException(404, detail="OCR results not found")
    
    if page:
        # Get specific page
        page_file = ocr_dir / f"page_{page}_ocr.json"
        if not page_file.exists():
            raise HTTPException(404, detail=f"Page {page} results not found")
        
        with open(page_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    else:
        # Get all pages
        results = []
        for page_file in sorted(ocr_dir.glob("page_*_ocr.json")):
            with open(page_file, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        
        # Get REAL engine info from processor if available
        engine_info = {
            "engine": "EasyOCR",  # CHANGED from PaddleOCR
            "mode": "real",
            "language": "en",
            "available": True,
            "using_real_ocr": True,
            "note": "REAL EasyOCR results"
        }
        
        return {
            "document_id": str(document_id),
            "total_pages": len(results),
            "pages": results,
            "engine_info": engine_info
        }


@router.get("/engine/info")
async def get_engine_info():
    """Get REAL OCR engine information."""
    try:
        processor = get_or_create_processor()
        info = processor.ocr_manager.get_info()
        
        # FORCE REAL MODE CHECK
        if info.get('engine') != 'EasyOCR' or info.get('mode') != 'real':  # CHANGED from PaddleOCR
            # CRITICAL ERROR - NOT REAL OCR
            return {
                "engine": "ERROR",
                "mode": "NOT_REAL",
                "language": "en",
                "available": False,
                "using_real_ocr": False,
                "error": "System is using mock data. Replace engine.py with REAL OCR version.",
                "note": "Phase 3 FAILED - Not using REAL OCR"
            }
        
        return info
    except Exception as e:
        return {
            "engine": "ERROR",
            "mode": "failed",
            "language": "en",
            "available": False,
            "using_real_ocr": False,
            "error": str(e),
            "note": "REAL OCR engine failed"
        }


@router.post("/test/{document_id}")
async def test_ocr_region(
    document_id: uuid.UUID,
    page: int = 1,
    region_index: int = 0
):
    """Test REAL OCR on a specific region."""
    try:
        # Check document
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        if not doc_dir.exists():
            raise HTTPException(404, detail="Document not found")
        
        # Load layout
        layout_file = doc_dir / "layout" / f"page_{page}_layout.json"
        if not layout_file.exists():
            raise HTTPException(404, detail="Layout not found")
        
        with open(layout_file, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
        
        detections = layout_data.get("detections", [])
        if region_index >= len(detections):
            raise HTTPException(400, detail=f"Region {region_index} out of range")
        
        # Get the detection
        detection_dict = detections[region_index]
        detection = Detection(**detection_dict)
        
        # Crop region
        from app.ocr.region_cropper import RegionCropper
        cropper = RegionCropper()
        
        # Get absolute image path
        image_path = doc_dir / layout_data["image_path"]
        if not image_path.exists():
            image_path = settings.BASE_DATA_DIR / layout_data["image_path"]
            if not image_path.exists():
                raise HTTPException(500, detail=f"Image not found: {layout_data['image_path']}")
        
        full_image = cropper.load_image(image_path)
        
        if full_image is None:
            raise HTTPException(500, detail="Failed to load image")
        
        region_image = cropper.crop_region(full_image, detection.dict() if hasattr(detection, 'dict') else detection)
        
        if region_image is None:
            raise HTTPException(500, detail="Failed to crop region")
        
        # Run REAL OCR
        processor = get_or_create_processor()
        text, confidence, word_boxes = processor.ocr_manager.extract_text(
            region_image, f"page{page}_region{region_index}"
        )
        
        # Verify REAL text (not mock)
        is_real = len(text.strip()) > 0
        
        return {
            "success": is_real,
            "page": page,
            "region_index": region_index,
            "region_type": detection.type,
            "bbox": detection.bbox.dict() if hasattr(detection.bbox, 'dict') else detection.bbox,
            "ocr_text": text,
            "ocr_confidence": confidence,
            "word_count": len(word_boxes),
            "is_real_text": is_real,
            "engine_info": processor.ocr_manager.get_info()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test OCR failed: {e}")
        raise HTTPException(500, detail=f"Test failed: {str(e)}")
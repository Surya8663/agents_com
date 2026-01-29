"""
OCR API endpoints for Phase 3 - REAL EasyOCR ONLY.
"""
import uuid
import json
import logging
import time
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.config.settings import settings
from app.ocr.processor import OCRProcessor
from app.vision.schema import Detection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])


def _find_layout_files(layout_dir: Path) -> List[Path]:
    """
    Find layout files with flexible pattern matching.
    Handles different naming patterns including typos.
    """
    if not layout_dir.exists():
        return []
    
    all_files = list(layout_dir.glob("*.json"))
    logger.debug(f"Found {len(all_files)} JSON files in {layout_dir}")
    
    # Priority 1: Exact pattern page_*_layout.json
    exact_files = []
    for f in all_files:
        name = f.name.lower()
        if name.startswith("page_") and name.endswith("_layout.json"):
            exact_files.append(f)
    
    if exact_files:
        exact_files.sort(key=lambda x: x.name)
        logger.info(f"Found {len(exact_files)} exact layout files")
        return exact_files
    
    # Priority 2: Flexible pattern page_*_layout*.json (handles typos)
    flexible_files = []
    for f in all_files:
        name = f.name.lower()
        if "page_" in name and "layout" in name:
            flexible_files.append(f)
    
    if flexible_files:
        flexible_files.sort(key=lambda x: x.name)
        logger.info(f"Found {len(flexible_files)} flexible layout files")
        return flexible_files
    
    # Priority 3: Any page_*.json files
    page_files = []
    for f in all_files:
        if "page_" in f.name.lower():
            page_files.append(f)
    
    if page_files:
        page_files.sort(key=lambda x: x.name)
        logger.info(f"Found {len(page_files)} page files")
        return page_files
    
    # Priority 4: Check for document_layout_summary.json
    summary_file = layout_dir / "document_layout_summary.json"
    if summary_file.exists():
        logger.info("Found document_layout_summary.json")
        return [summary_file]
    
    # Last resort: all JSON files
    if all_files:
        all_files.sort(key=lambda x: x.name)
        logger.info(f"Using all {len(all_files)} JSON files as fallback")
        return all_files
    
    return []


def _extract_page_number_from_filename(filename: str, index: int) -> int:
    """Extract page number from filename, fallback to index."""
    try:
        # Try patterns like: page_1_layout.json, page_1_layoutt.json, page1_layout.json
        name = filename.lower().replace("page_", "page").replace("_layout", "").replace(".json", "")
        if name.startswith("page"):
            num_part = name[4:]  # Remove "page"
            if num_part:
                # Extract digits
                digits = ''.join(filter(str.isdigit, num_part))
                if digits:
                    return int(digits)
    except:
        pass
    
    # Fallback to 1-based index
    return index + 1


async def _process_document_ocr(document_id: uuid.UUID, processor: OCRProcessor):
    """Background task to process REAL OCR."""
    try:
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        layout_dir = doc_dir / "layout"
        
        # Create OCR directory FIRST to track progress
        ocr_dir = doc_dir / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created OCR directory: {ocr_dir}")
        
        # Find layout files with flexible pattern matching
        layout_files = _find_layout_files(layout_dir)
        total_pages = len(layout_files)
        
        if total_pages == 0:
            logger.error(f"No layout files found in {layout_dir}")
            # List what files actually exist
            if layout_dir.exists():
                existing_files = list(layout_dir.glob("*"))
                logger.error(f"Directory exists. Files in layout dir: {[f.name for f in existing_files]}")
            else:
                logger.error(f"Layout directory doesn't exist: {layout_dir}")
            
            # Create error marker
            with open(ocr_dir / "ocr_failed.txt", "w") as f:
                f.write(f"No layout files found. Checked {layout_dir}")
            return
        
        logger.info(f"Found {total_pages} layout files: {[f.name for f in layout_files[:5]]}")
        if total_pages > 5:
            logger.info(f"... and {total_pages - 5} more files")
        
        processed_pages = 0
        failed_pages = 0
        
        for i, page_file in enumerate(layout_files):
            try:
                logger.info(f"Processing page {i+1}/{total_pages}: {page_file.name}")
                with open(page_file, "r", encoding="utf-8") as f:
                    page_data = json.load(f)
                
                # Ensure page_data has page_number
                if "page_number" not in page_data:
                    page_number = _extract_page_number_from_filename(page_file.name, i)
                    page_data["page_number"] = page_number
                    logger.info(f"Added page_number {page_number} to {page_file.name}")
                
                # Ensure page_data has image_path
                if "image_path" not in page_data:
                    # Try to construct image path
                    pages_dir = doc_dir / "pages"
                    page_num = page_data.get("page_number", i + 1)
                    image_path = f"pages/page_{page_num}.png"
                    if (doc_dir / image_path).exists():
                        page_data["image_path"] = image_path
                        logger.info(f"Added image_path {image_path} to {page_file.name}")
                
                # Process the page with OCR
                result = processor.process_page(document_id, page_data)
                if result:
                    processed_pages += 1
                    logger.info(f"✅ Page {i+1}/{total_pages}: {result.ocr_regions} regions with REAL text")
                    
                    # Small delay to prevent resource exhaustion
                    if i < total_pages - 1:
                        time.sleep(0.1)
                else:
                    failed_pages += 1
                    logger.warning(f"⚠️ Page {i+1}/{total_pages}: No OCR result")
                
            except Exception as e:
                failed_pages += 1
                logger.error(f"❌ Error processing {page_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Log final results
        if processed_pages > 0:
            success_rate = (processed_pages / total_pages) * 100
            logger.info(f"✅ OCR complete for {document_id}: {processed_pages}/{total_pages} pages processed ({success_rate:.1f}% success)")
            
            # Create completion marker
            with open(ocr_dir / "ocr_completed.txt", "w") as f:
                f.write(f"OCR completed: {processed_pages}/{total_pages} pages\n")
                f.write(f"Success rate: {success_rate:.1f}%\n")
                f.write(f"Failed: {failed_pages} pages\n")
        else:
            logger.error(f"❌ OCR failed for {document_id}: 0/{total_pages} pages processed")
            
            # Create error marker
            with open(ocr_dir / "ocr_failed.txt", "w") as f:
                f.write(f"OCR failed: 0/{total_pages} pages processed\n")
                f.write("All pages failed or no text extracted\n")
        
    except Exception as e:
        logger.error(f"❌ OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error marker
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        ocr_dir = doc_dir / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        with open(ocr_dir / "ocr_failed.txt", "w") as f:
            f.write(f"OCR processing failed: {str(e)}\n")


@router.post("/process/{document_id}")
async def process_document_ocr(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    lang: Optional[str] = "en",
    use_gpu: Optional[bool] = False
):
    """
    Start REAL OCR processing for a document.
    
    This endpoint:
    1. Checks if document and layout exist
    2. Initializes REAL EasyOCR processor
    3. Starts background OCR processing
    4. Returns immediate response with status
    """
    try:
        # Check document exists
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Check layout exists
        layout_dir = doc_dir / "layout"
        if not layout_dir.exists():
            raise HTTPException(status_code=400, detail="Run layout analysis first (/layout/analyze)")
        
        # Check if layout has any files
        layout_files = _find_layout_files(layout_dir)
        if not layout_files:
            # List what files actually exist
            existing_files = list(layout_dir.glob("*"))
            file_list = [f.name for f in existing_files]
            raise HTTPException(
                status_code=400, 
                detail=f"Layout analysis produced no valid files. Found: {file_list}"
            )
        
        # Create REAL OCR processor
        logger.info(f"🔄 Initializing OCR processor for document {document_id}")
        try:
            processor = OCRProcessor(lang=lang, use_gpu=use_gpu)
            engine_info = processor.ocr_manager.get_info()
            
            # Verify it's REAL
            if engine_info.get('mode') != 'real':
                raise HTTPException(
                    status_code=500, 
                    detail="OCR engine is not in REAL mode. Check EasyOCR installation."
                )
            
            logger.info(f"✅ OCR processor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ OCR processor initialization failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"OCR processor initialization failed: {str(e)}"
            )
        
        # Start processing in background
        background_tasks.add_task(_process_document_ocr, document_id, processor)
        
        return {
            "message": "REAL OCR processing started",
            "document_id": str(document_id),
            "engine": "EasyOCR",
            "mode": "real",
            "language": lang,
            "gpu_enabled": use_gpu,
            "layout_files_found": len(layout_files),
            "layout_file_names": [f.name for f in layout_files[:3]],  # First 3 names
            "engine_info": engine_info,
            "note": "OCR processing runs in background. Check /ocr/status/{id} for progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start OCR: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"OCR initialization failed: {str(e)}"
        )


@router.get("/status/{document_id}")
async def get_ocr_status(document_id: uuid.UUID):
    """
    Get REAL OCR processing status.
    
    Returns detailed status including:
    - Processing state
    - Files processed count
    - Success/failure information
    """
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    ocr_dir = doc_dir / "ocr"
    
    # Check for error/completion markers first
    if ocr_dir.exists():
        error_file = ocr_dir / "ocr_failed.txt"
        if error_file.exists():
            try:
                with open(error_file, "r") as f:
                    error_msg = f.read()
                return {
                    "status": "failed",
                    "document_id": str(document_id),
                    "error": error_msg,
                    "engine": "EasyOCR",
                    "mode": "real"
                }
            except:
                pass
        
        completion_file = ocr_dir / "ocr_completed.txt"
        if completion_file.exists():
            try:
                with open(completion_file, "r") as f:
                    completion_msg = f.read()
            except:
                completion_msg = "OCR completed"
    
    # If OCR directory doesn't exist, processing hasn't started
    if not ocr_dir.exists():
        return {
            "status": "not_started",
            "document_id": str(document_id),
            "engine": "EasyOCR",
            "mode": "real",
            "message": "OCR processing not started. Run /ocr/process first."
        }
    
    # Count OCR result files
    ocr_files = list(ocr_dir.glob("page_*_ocr.json"))
    
    # Count expected layout files
    layout_dir = doc_dir / "layout"
    layout_files = _find_layout_files(layout_dir)
    
    if not ocr_files:
        # Check if any processing markers exist
        if ocr_dir.exists() and any(ocr_dir.glob("*")):
            return {
                "status": "processing",
                "document_id": str(document_id),
                "engine": "EasyOCR",
                "mode": "real",
                "layout_files": len(layout_files),
                "ocr_files": 0,
                "message": "OCR is processing. No result files yet."
            }
        else:
            return {
                "status": "not_started",
                "document_id": str(document_id),
                "engine": "EasyOCR",
                "mode": "real",
                "message": "OCR directory exists but no processing started."
            }
    
    # Count REAL text results and gather statistics
    real_text_pages = 0
    total_regions = 0
    ocr_regions_total = 0
    sample_texts = []
    
    for ocr_file in ocr_files[:3]:  # Check first 3 files for samples
        try:
            with open(ocr_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("ocr_regions", 0) > 0:
                    real_text_pages += 1
                total_regions += data.get("total_regions", 0)
                ocr_regions_total += data.get("ocr_regions", 0)
                
                # Get sample text from first region if available
                regions = data.get("regions", [])
                if regions and len(regions) > 0:
                    region = regions[0]
                    text = region.get("ocr_text", "")
                    if text and len(text.strip()) > 0:
                        sample_texts.append(text[:50] + "..." if len(text) > 50 else text)
        except Exception as e:
            logger.warning(f"Could not read OCR file {ocr_file.name}: {e}")
    
    # Determine status
    status = "completed" if len(ocr_files) >= len(layout_files) else "processing"
    
    response = {
        "status": status,
        "document_id": str(document_id),
        "pages_processed": len(ocr_files),
        "total_pages_expected": len(layout_files),
        "pages_with_real_text": real_text_pages,
        "total_regions": total_regions,
        "ocr_regions": ocr_regions_total,
        "success_rate": f"{(real_text_pages / len(ocr_files) * 100):.1f}%" if ocr_files else "0%",
        "engine": "EasyOCR",
        "mode": "real"
    }
    
    # Add sample text if available
    if sample_texts:
        response["sample_texts"] = sample_texts
    
    # Add completion message if available
    if 'completion_msg' in locals():
        response["completion_message"] = completion_msg
    
    return response


@router.get("/results/{document_id}")
async def get_ocr_results(
    document_id: uuid.UUID,
    page: Optional[int] = None
):
    """Get REAL OCR results."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    ocr_dir = doc_dir / "ocr"
    
    if not ocr_dir.exists():
        raise HTTPException(status_code=404, detail="OCR results not found. Run /ocr/process first.")
    
    # Check for error marker
    error_file = ocr_dir / "ocr_failed.txt"
    if error_file.exists():
        with open(error_file, "r") as f:
            error_msg = f.read()
        raise HTTPException(status_code=400, detail=f"OCR failed: {error_msg}")
    
    if page:
        # Get specific page
        page_file = ocr_dir / f"page_{page}_ocr.json"
        if not page_file.exists():
            raise HTTPException(status_code=404, detail=f"Page {page} results not found")
        
        with open(page_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    else:
        # Get all pages
        results = []
        ocr_files = sorted(ocr_dir.glob("page_*_ocr.json"), 
                          key=lambda x: int(x.stem.split('_')[1]))
        
        for page_file in ocr_files:
            try:
                with open(page_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.error(f"Could not read OCR file {page_file.name}: {e}")
                continue
        
        if not results:
            raise HTTPException(status_code=404, detail="No valid OCR results found")
        
        # Get REAL engine info
        engine_info = {
            "engine": "EasyOCR",
            "mode": "real",
            "language": "en",
            "available": True,
            "using_real_ocr": True,
            "note": "REAL EasyOCR results - No mock data"
        }
        
        return {
            "document_id": str(document_id),
            "total_pages": len(results),
            "pages": results,
            "engine_info": engine_info,
            "statistics": {
                "total_regions": sum(len(page.get("regions", [])) for page in results),
                "ocr_regions": sum(page.get("ocr_regions", 0) for page in results),
                "skipped_regions": sum(page.get("skipped_regions", 0) for page in results)
            }
        }


@router.get("/engine/info")
async def get_engine_info():
    """Get REAL OCR engine information."""
    try:
        # Try to initialize OCR processor to get real info
        processor = OCRProcessor(lang="en", use_gpu=False)
        info = processor.ocr_manager.get_info()
        
        # Verify it's REAL
        if info.get('engine') != 'EasyOCR' or info.get('mode') != 'real':
            return {
                "engine": "ERROR",
                "mode": "NOT_REAL",
                "language": "en",
                "available": False,
                "using_real_ocr": False,
                "error": "System is using mock data. Check app/ocr/engine.py",
                "note": "Phase 3 FAILED - Not using REAL OCR"
            }
        
        return info
        
    except Exception as e:
        logger.error(f"OCR engine info failed: {e}")
        return {
            "engine": "ERROR",
            "mode": "failed",
            "language": "en",
            "available": False,
            "using_real_ocr": False,
            "error": str(e),
            "note": "REAL OCR engine initialization failed"
        }


@router.get("/test/simple")
async def test_simple_ocr():
    """Simple test to verify OCR works."""
    try:
        # Create a simple test image with text
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        logger.info("🔄 Creating test image for OCR...")
        
        # Create a blank image
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            logger.debug("Using Arial font")
        except:
            font = ImageFont.load_default()
            logger.debug("Using default font")
        
        draw.text((10, 40), "TEST OCR WORKING 123", fill='black', font=font)
        
        # Convert to numpy array
        test_image = np.array(img)
        logger.debug(f"Test image shape: {test_image.shape}")
        
        # Initialize OCR processor
        logger.info("🔄 Initializing OCR processor for test...")
        processor = OCRProcessor(lang="en", use_gpu=False)
        
        # Test OCR extraction
        logger.info("🔄 Testing OCR extraction...")
        text, confidence, word_boxes = processor.ocr_manager.extract_text(test_image, "test_image")
        
        result = {
            "success": len(text.strip()) > 0,
            "text": text,
            "confidence": confidence,
            "word_count": len(word_boxes),
            "engine": "EasyOCR",
            "mode": "real"
        }
        
        if text:
            logger.info(f"✅ OCR test successful: '{text}' (confidence: {confidence:.2f})")
            result["note"] = "OCR is working correctly"
        else:
            logger.warning(f"⚠️ OCR test extracted no text")
            result["note"] = "OCR extracted no text from test image"
        
        return result
        
    except ImportError as e:
        logger.error(f"❌ Import error in OCR test: {e}")
        return {
            "success": False,
            "error": f"Import error: {str(e)}",
            "engine": "EasyOCR",
            "note": "Required packages missing. Install: pip install easyocr pillow numpy"
        }
    except Exception as e:
        logger.error(f"❌ OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "engine": "EasyOCR",
            "note": "OCR test failed"
        }


@router.get("/debug/files/{document_id}")
async def debug_ocr_files(document_id: uuid.UUID):
    """Debug endpoint to see OCR-related files."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    result = {
        "document_id": str(document_id),
        "base_dir": str(doc_dir.absolute())
    }
    
    # Check layout directory
    layout_dir = doc_dir / "layout"
    if layout_dir.exists():
        layout_files = list(layout_dir.glob("*"))
        json_files = list(layout_dir.glob("*.json"))
        
        result["layout"] = {
            "exists": True,
            "total_files": len(layout_files),
            "json_files": len(json_files),
            "json_file_names": [f.name for f in json_files],
            "all_files": [f.name for f in layout_files]
        }
        
        # Show first JSON file structure
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    sample_data = json.load(f)
                    result["layout"]["sample_file_structure"] = {
                        "file": json_files[0].name,
                        "keys": list(sample_data.keys()),
                        "has_page_number": "page_number" in sample_data,
                        "has_image_path": "image_path" in sample_data,
                        "has_detections": "detections" in sample_data,
                        "detections_count": len(sample_data.get("detections", [])) if "detections" in sample_data else 0
                    }
            except Exception as e:
                result["layout"]["sample_error"] = str(e)
    else:
        result["layout"] = {"exists": False}
    
    # Check OCR directory
    ocr_dir = doc_dir / "ocr"
    if ocr_dir.exists():
        ocr_files = list(ocr_dir.glob("*"))
        ocr_json_files = list(ocr_dir.glob("*.json"))
        marker_files = [f.name for f in ocr_files if f.name.endswith(".txt")]
        
        result["ocr"] = {
            "exists": True,
            "total_files": len(ocr_files),
            "ocr_json_files": len(ocr_json_files),
            "marker_files": marker_files,
            "ocr_file_names": [f.name for f in ocr_json_files],
            "all_files": [f.name for f in ocr_files]
        }
        
        # Read marker files if they exist
        for marker in ["ocr_failed.txt", "ocr_completed.txt"]:
            marker_path = ocr_dir / marker
            if marker_path.exists():
                try:
                    with open(marker_path, "r") as f:
                        result["ocr"][marker] = f.read()
                except:
                    result["ocr"][f"{marker}_error"] = "Could not read"
    else:
        result["ocr"] = {"exists": False}
    
    return result
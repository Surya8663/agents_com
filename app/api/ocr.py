"""
OCR API - Phase 3
FINAL FIX - Waits for layout completion before processing
"""
import uuid
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
import asyncio

from app.config.settings import settings
from app.ocr.processor import OCRProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])


def _wait_for_layout_completion(layout_dir: Path, timeout_seconds: int = 60) -> bool:
    """
    Wait for layout processing to complete.
    Returns True if completed, False if timeout.
    """
    processing_file = layout_dir / ".processing"
    start_time = time.time()
    
    logger.info(f"Waiting for layout processing to complete...")
    
    while (time.time() - start_time) < timeout_seconds:
        if not processing_file.exists():
            # Check if we have actual layout files
            json_files = list(layout_dir.glob("*.json"))
            if json_files:
                logger.info(f"✅ Layout processing completed! Found {len(json_files)} layout files")
                return True
            else:
                logger.warning("Processing file removed but no layout files found")
                return False
        
        # Wait a bit before checking again
        time.sleep(1)
        
        # Log progress every 5 seconds
        elapsed = int(time.time() - start_time)
        if elapsed % 5 == 0 and elapsed > 0:
            logger.info(f"Still waiting for layout... ({elapsed}s elapsed)")
    
    logger.warning(f"⚠️ Timeout waiting for layout processing after {timeout_seconds}s")
    return False


def _load_and_normalize_layout(layout_dir: Path, wait_for_completion: bool = True) -> List[Dict[str, Any]]:
    """
    Load and normalize layout data from any available format.
    Returns list of normalized page layouts.
    """
    if not layout_dir.exists():
        logger.error(f"Layout directory does not exist: {layout_dir}")
        return []
    
    # Check if processing is still ongoing
    processing_file = layout_dir / ".processing"
    if processing_file.exists():
        if wait_for_completion:
            logger.info("Layout is still processing. Waiting for completion...")
            if not _wait_for_layout_completion(layout_dir, timeout_seconds=60):
                logger.error("Layout processing did not complete in time")
                return []
        else:
            logger.warning("Layout is still processing and wait_for_completion=False")
            return []
    
    logger.info(f"Looking for layout files in: {layout_dir}")
    
    # Look for all possible layout files
    possible_files = [
        layout_dir / "document_layout_summary.json",  # Layout phase output
        layout_dir / "layout.json",                   # Expected format
        layout_dir / "layout_data.json",              # Alternative format
    ]
    
    # Add any page_*_layout.json files
    page_files = list(layout_dir.glob("page_*_layout.json"))
    logger.info(f"Found {len(page_files)} page layout files: {[f.name for f in page_files]}")
    
    if page_files:
        possible_files.extend(sorted(page_files, 
            key=lambda x: int(x.stem.split('_')[1]) if len(x.stem.split('_')) > 1 and x.stem.split('_')[1].isdigit() else 0))
    
    logger.info(f"Checking {len(possible_files)} possible files: {[f.name for f in possible_files]}")
    
    for file_path in possible_files:
        logger.info(f"Checking if file exists: {file_path}")
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                logger.info(f"✅ Found layout file: {file_path.name}")
                logger.info(f"File structure keys: {list(data.keys())}")
                
                # Normalize the data structure
                normalized = _normalize_layout_data(data, file_path.name, layout_dir)
                logger.info(f"✅ Normalized {len(normalized)} pages from {file_path.name}")
                return normalized
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    logger.warning(f"No valid layout files found in {layout_dir}")
    # List all files that do exist
    all_files = list(layout_dir.glob("*"))
    logger.warning(f"All files in layout directory: {[f.name for f in all_files]}")
    return []


def _normalize_image_path(image_path: str, page_number: int) -> str:
    """
    Normalize image path from layout to be relative to document directory.
    
    Layout saves paths as: documents/{doc_id}/pages/page_1.png
    OCR needs: pages/page_1.png
    """
    if not image_path:
        return f"pages/page_{page_number}.png"
    
    # If path contains "documents/" extract relative path after pages/
    if "documents/" in image_path and "/pages/" in image_path:
        # Extract just "pages/page_X.png"
        normalized = "pages/" + image_path.split("/pages/")[-1]
        logger.info(f"Normalized image path: {image_path} -> {normalized}")
        return normalized
    
    # Handle Windows paths
    if "documents\\" in image_path and "\\pages\\" in image_path:
        normalized = "pages/" + image_path.split("\\pages\\")[-1].replace("\\", "/")
        logger.info(f"Normalized Windows image path: {image_path} -> {normalized}")
        return normalized
    
    # If path already starts with pages/ use as-is
    if image_path.startswith("pages/") or image_path.startswith("pages\\"):
        return image_path.replace("\\", "/")
    
    # Default fallback
    return f"pages/page_{page_number}.png"


def _normalize_layout_data(data: Dict[str, Any], source_file: str, layout_dir: Path) -> List[Dict[str, Any]]:
    """
    Normalize layout data from different formats to OCR-compatible format.
    """
    normalized_pages = []
    
    # Case 1: document_layout_summary.json format (from layout phase)
    if "pages" in data and isinstance(data["pages"], list):
        logger.info(f"Normalizing document_layout_summary.json format from {source_file}")
        
        for page_idx, page_data in enumerate(data["pages"]):
            # Extract page number
            page_number = page_data.get("page_number", page_idx + 1)
            
            # Normalize detections to regions/boxes
            detections = page_data.get("detections", [])
            regions = []
            
            for det_idx, detection in enumerate(detections):
                # Convert detection to OCR-compatible region
                region = {
                    "id": f"region_{page_number}_{det_idx}",
                    "type": detection.get("label", "text"),
                    "bbox": detection.get("bbox", [0, 0, 100, 100]),  # x1, y1, x2, y2
                    "confidence": detection.get("confidence", 0.9),
                    "original_label": detection.get("label", "unknown"),
                    "page_number": page_number,
                }
                regions.append(region)
            
            # FIX: Normalize image path
            raw_image_path = page_data.get("image_path", f"pages/page_{page_number}.png")
            image_path = _normalize_image_path(raw_image_path, page_number)
            
            # Create normalized page
            normalized_page = {
                "page_number": page_number,
                "image_path": image_path,
                "page_width": page_data.get("page_width", 595),
                "page_height": page_data.get("page_height", 842),
                "regions": regions,
                "total_regions": len(regions),
                "source_format": "document_layout_summary",
                "source_file": source_file,
            }
            normalized_pages.append(normalized_page)
        
        logger.info(f"Normalized {len(normalized_pages)} pages from document_layout_summary.json")
        return normalized_pages
    
    # Case 2: Direct layout.json format
    elif "regions" in data or "blocks" in data or "bounding_boxes" in data:
        logger.info(f"Using direct layout format from {source_file}")
        
        if isinstance(data, list):
            for page_idx, page_data in enumerate(data):
                page_number = page_data.get("page_number", page_idx + 1)
                raw_image_path = page_data.get("image_path", f"pages/page_{page_number}.png")
                image_path = _normalize_image_path(raw_image_path, page_number)
                
                normalized_page = {
                    "page_number": page_number,
                    "image_path": image_path,
                    "regions": page_data.get("regions", page_data.get("blocks", page_data.get("bounding_boxes", []))),
                    "total_regions": len(page_data.get("regions", page_data.get("blocks", page_data.get("bounding_boxes", [])))),
                    "source_format": "direct_layout",
                    "source_file": source_file,
                }
                normalized_pages.append(normalized_page)
        else:
            page_number = data.get("page_number", 1)
            raw_image_path = data.get("image_path", f"pages/page_{page_number}.png")
            image_path = _normalize_image_path(raw_image_path, page_number)
            
            normalized_page = {
                "page_number": page_number,
                "image_path": image_path,
                "regions": data.get("regions", data.get("blocks", data.get("bounding_boxes", []))),
                "total_regions": len(data.get("regions", data.get("blocks", data.get("bounding_boxes", [])))),
                "source_format": "direct_layout",
                "source_file": source_file,
            }
            normalized_pages.append(normalized_page)
        
        return normalized_pages
    
    # Case 3: Single page_*_layout.json file
    elif source_file.startswith("page_") and source_file.endswith("_layout.json"):
        logger.info(f"Processing single page layout file: {source_file}")
        
        try:
            page_num_str = source_file.replace("page_", "").replace("_layout.json", "")
            page_number = int(page_num_str) if page_num_str.isdigit() else 1
        except:
            page_number = 1
        
        if "detections" in data:
            detections = data.get("detections", [])
            regions = []
            
            for det_idx, detection in enumerate(detections):
                region = {
                    "id": f"region_{page_number}_{det_idx}",
                    "type": detection.get("label", "text"),
                    "bbox": detection.get("bbox", [0, 0, 100, 100]),
                    "confidence": detection.get("confidence", 0.9),
                    "original_label": detection.get("label", "unknown"),
                    "page_number": page_number,
                }
                regions.append(region)
            
            raw_image_path = data.get("image_path", f"pages/page_{page_number}.png")
            image_path = _normalize_image_path(raw_image_path, page_number)
            
            normalized_page = {
                "page_number": page_number,
                "image_path": image_path,
                "page_width": data.get("page_width", 595),
                "page_height": data.get("page_height", 842),
                "regions": regions,
                "total_regions": len(regions),
                "source_format": "single_page_layout",
                "source_file": source_file,
            }
        else:
            raw_image_path = data.get("image_path", f"pages/page_{page_number}.png")
            image_path = _normalize_image_path(raw_image_path, page_number)
            
            normalized_page = {
                "page_number": page_number,
                "image_path": image_path,
                "regions": data.get("regions", data.get("blocks", data.get("bounding_boxes", []))),
                "total_regions": len(data.get("regions", data.get("blocks", data.get("bounding_boxes", [])))),
                "source_format": "single_page_ocr",
                "source_file": source_file,
            }
        
        normalized_pages.append(normalized_page)
        return normalized_pages
    
    # Case 4: Unknown format
    else:
        logger.warning(f"Unknown layout format in {source_file}, creating default layout")
        logger.warning(f"Data keys: {list(data.keys())}")
        
        pages_dir = layout_dir.parent / "pages"
        if pages_dir.exists():
            page_images = list(pages_dir.glob("*.png"))
            num_pages = len(page_images)
            logger.info(f"Found {num_pages} page images in {pages_dir}")
        else:
            num_pages = 1
            logger.warning(f"Pages directory not found: {pages_dir}")
        
        for page_num in range(1, num_pages + 1):
            normalized_page = {
                "page_number": page_num,
                "image_path": f"pages/page_{page_num}.png",
                "regions": [
                    {
                        "id": f"region_{page_num}_0",
                        "type": "full_page",
                        "bbox": [0, 0, 595, 842],
                        "confidence": 1.0,
                        "page_number": page_num,
                    }
                ],
                "total_regions": 1,
                "source_format": "default_fallback",
                "source_file": source_file,
            }
            normalized_pages.append(normalized_page)
        
        logger.info(f"Created default layout for {num_pages} pages")
        return normalized_pages


async def _process_normalized_ocr(document_id: uuid.UUID, processor: OCRProcessor, normalized_pages: List[Dict[str, Any]]):
    """Background task to process OCR with normalized layout data."""
    try:
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        
        ocr_dir = doc_dir / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created OCR directory: {ocr_dir}")
        
        total_pages = len(normalized_pages)
        processed_pages = 0
        failed_pages = 0
        
        for page_data in normalized_pages:
            try:
                page_number = page_data.get("page_number", 0)
                logger.info(f"Processing page {page_number}/{total_pages}")
                
                image_path = page_data.get("image_path", f"pages/page_{page_number}.png")
                full_image_path = doc_dir / image_path
                
                if not full_image_path.exists():
                    logger.error(f"Image not found at {full_image_path}")
                    
                    pages_dir = doc_dir / "pages"
                    if pages_dir.exists():
                        possible_images = list(pages_dir.glob(f"page_{page_number}.*"))
                        if possible_images:
                            actual_image = possible_images[0]
                            image_path = f"pages/{actual_image.name}"
                            page_data["image_path"] = image_path
                            logger.info(f"Found image at alternative path: {image_path}")
                        else:
                            logger.error(f"No images found for page {page_number}")
                            failed_pages += 1
                            continue
                    else:
                        logger.error(f"Pages directory does not exist: {pages_dir}")
                        failed_pages += 1
                        continue
                
                logger.info(f"Using image path: {image_path}")
                
                result = processor.process_page(document_id, page_data)
                if result:
                    processed_pages += 1
                    logger.info(f"✅ Page {page_number}/{total_pages}: {result.ocr_regions} regions with text")
                    
                    if processed_pages < total_pages:
                        time.sleep(0.1)
                else:
                    failed_pages += 1
                    logger.warning(f"⚠️ Page {page_number}/{total_pages}: No OCR result")
                
            except Exception as e:
                failed_pages += 1
                logger.error(f"❌ Error processing page {page_data.get('page_number', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
        
        if processed_pages > 0:
            success_rate = (processed_pages / total_pages) * 100
            logger.info(f"✅ OCR complete for {document_id}: {processed_pages}/{total_pages} pages ({success_rate:.1f}%)")
            
            with open(ocr_dir / "ocr_completed.txt", "w") as f:
                f.write(f"OCR completed: {processed_pages}/{total_pages} pages\n")
                f.write(f"Success rate: {success_rate:.1f}%\n")
                f.write(f"Failed: {failed_pages} pages\n")
        else:
            logger.error(f"❌ OCR failed for {document_id}: 0/{total_pages} pages")
            
            with open(ocr_dir / "ocr_failed.txt", "w") as f:
                f.write(f"OCR failed: 0/{total_pages} pages processed\n")
        
    except Exception as e:
        logger.error(f"❌ OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        
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
    use_gpu: Optional[bool] = False,
    wait_for_layout: Optional[bool] = True
):
    """
    Start OCR processing for a document.
    
    Args:
        document_id: Document ID
        lang: OCR language (default: en)
        use_gpu: Use GPU for OCR (default: False)
        wait_for_layout: Wait for layout processing to complete (default: True)
    """
    try:
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        layout_dir = doc_dir / "layout"
        if not layout_dir.exists():
            raise HTTPException(status_code=400, detail="Run layout analysis first (/layout/analyze)")
        
        # Check if layout is still processing
        processing_file = layout_dir / ".processing"
        if processing_file.exists():
            if wait_for_layout:
                logger.info("⏳ Layout is still processing. Waiting for completion...")
            else:
                raise HTTPException(
                    status_code=425, 
                    detail="Layout analysis is still processing. Please wait and try again, or set wait_for_layout=true"
                )
        
        logger.info(f"Loading layout data for document {document_id}")
        normalized_pages = _load_and_normalize_layout(layout_dir, wait_for_completion=wait_for_layout)
        
        if not normalized_pages:
            existing_files = list(layout_dir.glob("*"))
            file_list = [f.name for f in existing_files]
            logger.error(f"No valid layout data found. Files in layout dir: {file_list}")
            
            # Give more helpful error message
            if processing_file.exists():
                raise HTTPException(
                    status_code=425,
                    detail="Layout analysis is still processing. Please wait a moment and try again."
                )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Layout analysis completed but no valid layout files found. Files: {file_list}"
                )
        
        logger.info(f"✅ Successfully normalized {len(normalized_pages)} pages for OCR")
        logger.info(f"Sample image paths: {[p['image_path'] for p in normalized_pages[:3]]}")
        
        logger.info(f"🔄 Initializing OCR processor...")
        try:
            processor = OCRProcessor(lang=lang, use_gpu=use_gpu)
            engine_info = processor.ocr_manager.get_info()
            
            if engine_info.get('mode') != 'real':
                raise HTTPException(
                    status_code=500, 
                    detail="OCR engine is not in REAL mode. Check EasyOCR installation."
                )
            
            logger.info(f"✅ OCR processor initialized")
            
        except Exception as e:
            logger.error(f"❌ OCR processor initialization failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"OCR processor initialization failed: {str(e)}"
            )
        
        background_tasks.add_task(_process_normalized_ocr, document_id, processor, normalized_pages)
        
        return {
            "message": "OCR processing started",
            "document_id": str(document_id),
            "engine": "EasyOCR",
            "mode": "real",
            "language": lang,
            "gpu_enabled": use_gpu,
            "layout_pages_found": len(normalized_pages),
            "layout_sources": [p.get("source_file", "unknown") for p in normalized_pages[:3]],
            "sample_image_paths": [p["image_path"] for p in normalized_pages[:3]],
            "engine_info": engine_info,
            "note": "OCR processing runs in background. Check /ocr/status/{id} for progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start OCR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"OCR initialization failed: {str(e)}"
        )


@router.get("/status/{document_id}")
async def get_ocr_status(document_id: uuid.UUID):
    """Get OCR processing status."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    ocr_dir = doc_dir / "ocr"
    
    if ocr_dir.exists():
        error_file = ocr_dir / "ocr_failed.txt"
        if error_file.exists():
            try:
                with open(error_file, "r") as f:
                    error_msg = f.read()
                return {
                    "status": "failed",
                    "document_id": str(document_id),
                    "error": error_msg
                }
            except:
                pass
        
        completion_file = ocr_dir / "ocr_completed.txt"
        completion_msg = None
        if completion_file.exists():
            try:
                with open(completion_file, "r") as f:
                    completion_msg = f.read()
            except:
                pass
    
    if not ocr_dir.exists():
        return {
            "status": "not_started",
            "document_id": str(document_id),
            "message": "OCR processing not started"
        }
    
    ocr_files = list(ocr_dir.glob("page_*_ocr.json"))
    
    layout_dir = doc_dir / "layout"
    total_pages_expected = 0
    if layout_dir.exists():
        try:
            normalized_pages = _load_and_normalize_layout(layout_dir, wait_for_completion=False)
            total_pages_expected = len(normalized_pages)
        except:
            pass
    
    if not ocr_files:
        if ocr_dir.exists() and any(ocr_dir.glob("*")):
            return {
                "status": "processing",
                "document_id": str(document_id),
                "layout_pages_expected": total_pages_expected,
                "ocr_files": 0,
                "message": "OCR is processing"
            }
        else:
            return {
                "status": "not_started",
                "document_id": str(document_id),
                "message": "OCR not started"
            }
    
    real_text_pages = 0
    total_regions = 0
    ocr_regions_total = 0
    sample_texts = []
    
    for ocr_file in ocr_files[:3]:
        try:
            with open(ocr_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("ocr_regions", 0) > 0:
                    real_text_pages += 1
                total_regions += data.get("total_regions", 0)
                ocr_regions_total += data.get("ocr_regions", 0)
                
                regions = data.get("regions", [])
                if regions and len(regions) > 0:
                    text = regions[0].get("ocr_text", "")
                    if text and len(text.strip()) > 0:
                        sample_texts.append(text[:50] + "..." if len(text) > 50 else text)
        except Exception as e:
            logger.warning(f"Could not read OCR file {ocr_file.name}: {e}")
    
    status = "completed" if len(ocr_files) >= total_pages_expected else "processing"
    
    response = {
        "status": status,
        "document_id": str(document_id),
        "pages_processed": len(ocr_files),
        "total_pages_expected": total_pages_expected,
        "pages_with_text": real_text_pages,
        "total_regions": total_regions,
        "ocr_regions": ocr_regions_total,
        "success_rate": f"{(real_text_pages / len(ocr_files) * 100):.1f}%" if ocr_files else "0%"
    }
    
    if sample_texts:
        response["sample_texts"] = sample_texts
    
    if completion_msg:
        response["completion_message"] = completion_msg
    
    return response


@router.get("/results/{document_id}")
async def get_ocr_results(document_id: uuid.UUID, page: Optional[int] = None):
    """Get OCR results."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    ocr_dir = doc_dir / "ocr"
    
    if not ocr_dir.exists():
        raise HTTPException(status_code=404, detail="OCR results not found")
    
    error_file = ocr_dir / "ocr_failed.txt"
    if error_file.exists():
        with open(error_file, "r") as f:
            error_msg = f.read()
        raise HTTPException(status_code=400, detail=f"OCR failed: {error_msg}")
    
    if page:
        page_file = ocr_dir / f"page_{page}_ocr.json"
        if not page_file.exists():
            raise HTTPException(status_code=404, detail=f"Page {page} results not found")
        
        with open(page_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    else:
        results = []
        ocr_files = sorted(ocr_dir.glob("page_*_ocr.json"), 
                          key=lambda x: int(x.stem.split('_')[1]) if len(x.stem.split('_')) > 1 and x.stem.split('_')[1].isdigit() else 0)
        
        for page_file in ocr_files:
            try:
                with open(page_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.error(f"Could not read OCR file {page_file.name}: {e}")
                continue
        
        if not results:
            raise HTTPException(status_code=404, detail="No valid OCR results found")
        
        return {
            "document_id": str(document_id),
            "total_pages": len(results),
            "pages": results,
            "statistics": {
                "total_regions": sum(len(page.get("regions", [])) for page in results),
                "ocr_regions": sum(page.get("ocr_regions", 0) for page in results),
                "skipped_regions": sum(page.get("skipped_regions", 0) for page in results)
            }
        }


@router.get("/engine/info")
async def get_engine_info():
    """Get OCR engine information."""
    try:
        processor = OCRProcessor(lang="en", use_gpu=False)
        info = processor.ocr_manager.get_info()
        return info
    except Exception as e:
        logger.error(f"OCR engine info failed: {e}")
        return {
            "engine": "ERROR",
            "error": str(e)
        }


@router.get("/test/simple")
async def test_simple_ocr():
    """Simple test to verify OCR works."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 40), "TEST OCR 123", fill='black', font=font)
        test_image = np.array(img)
        
        processor = OCRProcessor(lang="en", use_gpu=False)
        text, confidence, word_boxes = processor.ocr_manager.extract_text(test_image, "test")
        
        return {
            "success": len(text.strip()) > 0,
            "text": text,
            "confidence": confidence,
            "word_count": len(word_boxes)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/debug/files/{document_id}")
async def debug_ocr_files(document_id: uuid.UUID):
    """Debug endpoint."""
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document not found")
    
    result = {
        "document_id": str(document_id),
        "base_dir": str(doc_dir.absolute())
    }
    
    layout_dir = doc_dir / "layout"
    if layout_dir.exists():
        layout_files = list(layout_dir.glob("*"))
        json_files = list(layout_dir.glob("*.json"))
        
        result["layout"] = {
            "exists": True,
            "total_files": len(layout_files),
            "json_files": len(json_files),
            "json_file_names": [f.name for f in json_files],
            "all_files": [f.name for f in layout_files],
            "is_processing": (layout_dir / ".processing").exists()
        }
        
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    sample_data = json.load(f)
                    result["layout"]["sample"] = {
                        "file": json_files[0].name,
                        "keys": list(sample_data.keys())
                    }
            except Exception as e:
                result["layout"]["sample_error"] = str(e)
    else:
        result["layout"] = {"exists": False}
    
    ocr_dir = doc_dir / "ocr"
    if ocr_dir.exists():
        ocr_files = list(ocr_dir.glob("*"))
        result["ocr"] = {
            "exists": True,
            "total_files": len(ocr_files),
            "all_files": [f.name for f in ocr_files]
        }
    else:
        result["ocr"] = {"exists": False}
    
    return result
"""
Layout Analysis API - Phase 2
Complete implementation with proper routing
"""
import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional
import traceback

from fastapi import APIRouter, HTTPException, status, BackgroundTasks

from app.config.settings import settings
from app.vision.model_loader import ModelLoader
from app.vision.postprocessor import LayoutPostProcessor
from app.vision.schema import PageLayout, DocumentLayout

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/layout", tags=["layout"])


class LayoutAnalyzer:
    """Main orchestrator for layout analysis pipeline."""
    
    def __init__(self):
        """Initialize layout analyzer with model and processors."""
        self.model_loader = ModelLoader()
        self.model_loaded = self.model_loader.load()
        
        if not self.model_loaded:
            logger.warning("YOLO model failed to load. Layout detection may not work.")
        
        # Initialize detector
        try:
            from app.vision.enhanced_detector import EnhancedLayoutDetector
            self.detector = EnhancedLayoutDetector(self.model_loader)
        except ImportError:
            # Fallback to original detector
            from app.vision.detector import LayoutDetector
            self.detector = LayoutDetector(self.model_loader)
        
        # Initialize postprocessor
        self.postprocessor = LayoutPostProcessor()
        
        logger.info(f"LayoutAnalyzer initialized with model: {self.model_loader.model_path}")
    
    def analyze_document(self, document_id: uuid.UUID) -> Optional[DocumentLayout]:
        """
        Analyze layout for all pages of a document.
        
        Args:
            document_id: Document identifier from Phase 1
            
        Returns:
            DocumentLayout if successful, None otherwise
        """
        # Get paths
        doc_dir = settings.DOCUMENTS_DIR / str(document_id)
        pages_dir = doc_dir / "pages"
        layout_dir = doc_dir / "layout"
        
        # Create layout directory
        layout_dir.mkdir(parents=True, exist_ok=True)
        
        # Create processing file
        processing_file = layout_dir / ".processing"
        try:
            processing_file.touch()
        except:
            pass
        
        try:
            # Check if pages exist
            if not pages_dir.exists():
                logger.error(f"Pages directory not found for document {document_id}")
                self._create_empty_layout(document_id, layout_dir)
                return None
            
            # Get all page images
            page_images = sorted(
                pages_dir.glob("*.png"),
                key=lambda x: int(x.stem.split('_')[-1]) if x.stem.startswith("page_") else 0
            )
            
            if not page_images:
                logger.error(f"No page images found for document {document_id}")
                self._create_empty_layout(document_id, layout_dir)
                return None
            
            page_layouts = []
            
            # Process each page
            for page_num, image_path in enumerate(page_images, 1):
                logger.info(f"Processing page {page_num}: {image_path.name}")
                
                # Detect layout elements
                detections, (width, height) = self.detector.detect_page_layout(image_path)
                
                # Post-process detections
                cleaned_detections = self.postprocessor.process(detections)
                
                # Create page layout
                page_layout = PageLayout(
                    page_number=page_num,
                    image_path=str(image_path.relative_to(settings.BASE_DATA_DIR)),
                    page_width=width,
                    page_height=height,
                    detections=cleaned_detections
                )
                
                # Save individual page layout (for OCR compatibility)
                layout_path = layout_dir / f"page_{page_num}_layout.json"
                with open(layout_path, "w", encoding="utf-8") as f:
                    json_str = page_layout.model_dump_json(indent=2)
                    f.write(json_str)
                
                page_layouts.append(page_layout)
                
                logger.info(f"Page {page_num}: Found {len(cleaned_detections)} layout elements")
            
            # Create and save complete document layout
            document_layout = DocumentLayout(
                document_id=document_id,
                pages=page_layouts
            )
            
            # Save document layout summary (for future phases)
            summary_path = layout_dir / "document_layout_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json_str = document_layout.model_dump_json(indent=2)
                f.write(json_str)
            
            # Remove processing file on success
            if processing_file.exists():
                processing_file.unlink()
            
            logger.info(f"Layout analysis complete for document {document_id}")
            logger.info(f"Saved {len(page_layouts)} per-page layout files and document_layout_summary.json")
            return document_layout
            
        except Exception as e:
            logger.error(f"Error analyzing document {document_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Create error file
            try:
                error_file = layout_dir / ".error"
                with open(error_file, "w") as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(traceback.format_exc())
            except:
                pass
            
            # Remove processing file
            if processing_file.exists():
                processing_file.unlink()
            
            # Create empty layout as fallback
            self._create_empty_layout(document_id, layout_dir)
            return None
    
    def _create_empty_layout(self, document_id: uuid.UUID, layout_dir: Path):
        """Create empty layout files to prevent repeated failures."""
        try:
            # Create empty page layouts for each page found
            doc_dir = settings.DOCUMENTS_DIR / str(document_id)
            pages_dir = doc_dir / "pages"
            
            if pages_dir.exists():
                page_images = list(pages_dir.glob("*.png"))
                page_layouts = []
                
                for page_num, image_path in enumerate(page_images, 1):
                    # Create empty page layout
                    empty_layout = PageLayout(
                        page_number=page_num,
                        image_path=str(image_path.relative_to(settings.BASE_DATA_DIR)),
                        page_width=1,
                        page_height=1,
                        detections=[]
                    )
                    
                    # Save individual page layout file
                    layout_path = layout_dir / f"page_{page_num}_layout.json"
                    with open(layout_path, "w", encoding="utf-8") as f:
                        json_str = empty_layout.model_dump_json(indent=2)
                        f.write(json_str)
                    
                    page_layouts.append(empty_layout)
                
                # Create empty document summary
                empty_doc_layout = DocumentLayout(
                    document_id=document_id,
                    pages=page_layouts
                )
                
                summary_path = layout_dir / "document_layout_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json_str = empty_doc_layout.model_dump_json(indent=2)
                    f.write(json_str)
                    
                logger.info(f"Created empty layout for document {document_id} after failure")
                
        except Exception as e:
            logger.error(f"Failed to create empty layout: {e}")
    
    def get_layout_status(self, document_id: uuid.UUID) -> dict:
        """
        Check status of layout analysis for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with status information
        """
        layout_dir = settings.DOCUMENTS_DIR / str(document_id) / "layout"
        
        if not layout_dir.exists():
            return {"status": "not_started", "document_id": str(document_id)}
        
        # Check if analysis is in progress
        processing_file = layout_dir / ".processing"
        if processing_file.exists():
            return {"status": "processing", "document_id": str(document_id)}
        
        # Check for error file
        error_file = layout_dir / ".error"
        if error_file.exists():
            return {"status": "failed", "document_id": str(document_id), "error": "Analysis failed"}
        
        # Check for summary file
        summary_path = layout_dir / "document_layout_summary.json"
        if summary_path.exists():
            # Also check for per-page files
            layout_files = list(layout_dir.glob("page_*_layout.json"))
            return {
                "status": "completed", 
                "document_id": str(document_id),
                "has_summary_file": True,
                "has_per_page_files": len(layout_files) > 0,
                "per_page_files_count": len(layout_files)
            }
        
        # Check for individual page layouts
        layout_files = list(layout_dir.glob("page_*_layout.json"))
        if layout_files:
            return {
                "status": "partial",
                "document_id": str(document_id),
                "pages_processed": len(layout_files)
            }
        
        return {"status": "failed", "document_id": str(document_id)}


# Global analyzer instance
analyzer = LayoutAnalyzer()


@router.post("/analyze/{document_id}")
async def analyze_document_layout(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks
):
    """
    Trigger layout analysis for a document.
    
    This endpoint:
    1. Validates the document exists (from Phase 1)
    2. Starts background layout analysis
    3. Returns immediate status
    """
    # Check if document exists from Phase 1
    doc_dir = settings.DOCUMENTS_DIR / str(document_id)
    if not doc_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    
    # Check if already processed
    status_info = analyzer.get_layout_status(document_id)
    if status_info["status"] == "completed":
        return {
            "message": f"Document {document_id} already processed",
            "document_id": str(document_id),
            "status": "already_completed",
            "files_created": {
                "summary_file": status_info.get("has_summary_file", False),
                "per_page_files": status_info.get("has_per_page_files", False),
                "per_page_count": status_info.get("per_page_files_count", 0)
            }
        }
    
    # Add to background tasks
    background_tasks.add_task(analyzer.analyze_document, document_id)
    
    return {
        "message": f"Layout analysis started for document {document_id}",
        "document_id": str(document_id),
        "status": "processing_started",
        "note": "Layout analysis will create both per-page layout files and a document summary"
    }


@router.get("/status/{document_id}")
async def get_layout_status(document_id: uuid.UUID):
    """Get layout analysis status for a document."""
    status_info = analyzer.get_layout_status(document_id)
    return status_info


@router.get("/results/{document_id}")
async def get_layout_results(document_id: uuid.UUID, page: Optional[int] = None):
    """
    Get layout analysis results for a document.
    
    Args:
        document_id: Document identifier
        page: Optional specific page number
    """
    layout_dir = settings.DOCUMENTS_DIR / str(document_id) / "layout"
    
    if not layout_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Layout analysis not found for document {document_id}"
        )
    
    if page:
        # Return specific page layout
        layout_path = layout_dir / f"page_{page}_layout.json"
        if not layout_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Layout for page {page} not found"
            )
        
        with open(layout_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
        
        return {"document_id": str(document_id), "page": page, "layout": layout_data}
    
    else:
        # Return document summary
        summary_path = layout_dir / "document_layout_summary.json"
        if not summary_path.exists():
            # Fall back to individual page files
            layout_files = list(layout_dir.glob("page_*_layout.json"))
            if not layout_files:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No layout results found for document {document_id}"
                )
            
            # Compile from individual files
            all_layouts = []
            for layout_file in sorted(layout_files, key=lambda x: int(x.stem.split('_')[1])):
                with open(layout_file, "r", encoding="utf-8") as f:
                    all_layouts.append(json.load(f))
            
            return {
                "document_id": str(document_id),
                "status": "partial",
                "total_pages": len(all_layouts),
                "pages": all_layouts
            }
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        
        return summary_data


@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded YOLO model."""
    model_info = analyzer.model_loader.get_model_info()
    return model_info


@router.get("/debug/files/{document_id}")
async def debug_layout_files(document_id: uuid.UUID):
    """Debug endpoint to see layout files structure."""
    layout_dir = settings.DOCUMENTS_DIR / str(document_id) / "layout"
    
    if not layout_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Layout directory not found for document {document_id}"
        )
    
    result = {
        "document_id": str(document_id),
        "layout_dir": str(layout_dir.absolute()),
        "files": {}
    }
    
    # List all files
    all_files = list(layout_dir.glob("*"))
    result["total_files"] = len(all_files)
    
    # Categorize files
    json_files = list(layout_dir.glob("*.json"))
    page_files = list(layout_dir.glob("page_*_layout.json"))
    summary_files = list(layout_dir.glob("document_layout_summary.json"))
    other_files = list(layout_dir.glob("*"))
    
    # Remove json files from other files
    other_files = [f for f in other_files if f not in json_files]
    
    result["files"]["json_files"] = {
        "count": len(json_files),
        "names": [f.name for f in sorted(json_files)]
    }
    
    result["files"]["page_layout_files"] = {
        "count": len(page_files),
        "names": [f.name for f in sorted(page_files, key=lambda x: int(x.stem.split('_')[1]) if len(x.stem.split('_')) > 1 and x.stem.split('_')[1].isdigit() else 0)],
        "sample_structure": None
    }
    
    result["files"]["summary_file"] = {
        "exists": len(summary_files) > 0,
        "name": summary_files[0].name if summary_files else None
    }
    
    result["files"]["other_files"] = {
        "count": len(other_files),
        "names": [f.name for f in sorted(other_files)]
    }
    
    # Show sample structure if page files exist
    if page_files:
        try:
            with open(page_files[0], "r", encoding="utf-8") as f:
                sample_data = json.load(f)
                result["files"]["page_layout_files"]["sample_structure"] = {
                    "keys": list(sample_data.keys()),
                    "has_page_number": "page_number" in sample_data,
                    "page_number": sample_data.get("page_number"),
                    "has_image_path": "image_path" in sample_data,
                    "has_detections": "detections" in sample_data,
                    "detections_count": len(sample_data.get("detections", [])) if "detections" in sample_data else 0
                }
        except Exception as e:
            result["files"]["page_layout_files"]["sample_error"] = str(e)
    
    # Show summary structure if exists
    if summary_files:
        try:
            with open(summary_files[0], "r", encoding="utf-8") as f:
                summary_data = json.load(f)
                result["files"]["summary_file"]["structure"] = {
                    "keys": list(summary_data.keys()),
                    "has_pages": "pages" in summary_data,
                    "pages_count": len(summary_data.get("pages", [])) if "pages" in summary_data else 0
                }
        except Exception as e:
            result["files"]["summary_file"]["error"] = str(e)
    
    return result
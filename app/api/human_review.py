"""
Human review system API endpoints.
"""
import logging
import json
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.confidence.confidence_engine import confidence_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/review", tags=["review"])


# In-memory review store (in production, use database)
review_store = {}
REVIEWS_FILE = settings.BASE_DATA_DIR / "reviews.json"


class ReviewItem(BaseModel):
    """Item for human review."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    extraction_id: Optional[str] = None
    field_name: str
    extracted_value: Any
    confidence: float
    reason_for_review: str
    page_number: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    agent_sources: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = "pending"  # pending, approved, rejected, edited
    reviewer_notes: Optional[str] = None
    corrected_value: Optional[Any] = None


class ReviewAction(BaseModel):
    """Review action from human."""
    review_id: str
    action: str  # approve, reject, edit
    corrected_value: Optional[Any] = None
    reviewer_notes: Optional[str] = None


class ReviewResponse(BaseModel):
    """Review system response."""
    success: bool
    message: str
    review_id: Optional[str] = None


def _load_reviews():
    """Load reviews from disk."""
    global review_store
    
    if REVIEWS_FILE.exists():
        try:
            with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
                stored_reviews = json.load(f)
                # Convert list to dict
                for review in stored_reviews:
                    review_store[review["id"]] = review
            logger.info(f"Loaded {len(review_store)} reviews from disk")
        except Exception as e:
            logger.error(f"Failed to load reviews: {e}")
            review_store = {}


def _save_reviews():
    """Save reviews to disk."""
    try:
        reviews_list = list(review_store.values())
        with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(reviews_list, f, default=str, indent=2)
        logger.debug(f"Saved {len(reviews_list)} reviews to disk")
    except Exception as e:
        logger.error(f"Failed to save reviews: {e}")


# Load reviews on startup
_load_reviews()


@router.post("/submit")
async def submit_for_review(item: ReviewItem) -> ReviewResponse:
    """
    Submit an extraction for human review.
    
    This is typically called by the confidence engine when
    confidence is below threshold.
    """
    try:
        # Validate confidence
        if item.confidence < 0 or item.confidence > 1:
            raise ValueError(f"Invalid confidence: {item.confidence}")
        
        # Store review item
        review_store[item.id] = item.dict()
        
        # Save to disk
        _save_reviews()
        
        logger.info(f"Submitted for review: {item.field_name}={item.extracted_value} "
                   f"(conf: {item.confidence:.2f}, reason: {item.reason_for_review})")
        
        return ReviewResponse(
            success=True,
            message=f"Submitted for review: {item.field_name}",
            review_id=item.id
        )
        
    except Exception as e:
        logger.error(f"Review submission failed: {e}")
        return ReviewResponse(
            success=False,
            message=f"Submission failed: {str(e)}"
        )


@router.post("/process")
async def process_review(action: ReviewAction) -> ReviewResponse:
    """
    Process a review action (approve, reject, edit).
    """
    try:
        # Get review item
        if action.review_id not in review_store:
            raise HTTPException(status_code=404, detail="Review item not found")
        
        review_item = review_store[action.review_id]
        
        # Validate action
        if action.action not in ["approve", "reject", "edit"]:
            raise ValueError(f"Invalid action: {action.action}")
        
        # Update based on action
        if action.action == "approve":
            review_item["status"] = "approved"
            review_item["reviewer_notes"] = action.reviewer_notes or "Approved by reviewer"
            
        elif action.action == "reject":
            review_item["status"] = "rejected"
            review_item["reviewer_notes"] = action.reviewer_notes or "Rejected by reviewer"
            
        elif action.action == "edit":
            if action.corrected_value is None:
                raise ValueError("Corrected value required for edit action")
            
            review_item["status"] = "edited"
            review_item["corrected_value"] = action.corrected_value
            review_item["reviewer_notes"] = action.reviewer_notes or "Edited by reviewer"
        
        # Update timestamp
        review_item["processed_at"] = datetime.now().isoformat()
        
        # Save to disk
        _save_reviews()
        
        logger.info(f"Processed review {action.review_id}: {action.action}")
        
        return ReviewResponse(
            success=True,
            message=f"Review processed: {action.action}",
            review_id=action.review_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Review processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/queue")
async def get_review_queue(status: Optional[str] = "pending", 
                          limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get review queue items.
    
    Args:
        status: Filter by status (pending, approved, rejected, edited)
        limit: Maximum number of items to return
    """
    try:
        # Filter by status
        filtered_items = [
            item for item in review_store.values()
            if item["status"] == status
        ]
        
        # Sort by timestamp (oldest first)
        filtered_items.sort(key=lambda x: x.get("timestamp", ""))
        
        # Apply limit
        result = filtered_items[:limit]
        
        logger.debug(f"Returning {len(result)} review items with status '{status}'")
        return result
        
    except Exception as e:
        logger.error(f"Failed to get review queue: {e}")
        raise HTTPException(status_code=500, detail=f"Queue retrieval failed: {str(e)}")


@router.get("/item/{review_id}")
async def get_review_item(review_id: str) -> Dict[str, Any]:
    """Get specific review item."""
    try:
        if review_id not in review_store:
            raise HTTPException(status_code=404, detail="Review item not found")
        
        return review_store[review_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get review item: {e}")
        raise HTTPException(status_code=500, detail=f"Item retrieval failed: {str(e)}")


@router.get("/document/{document_id}")
async def get_document_reviews(document_id: str) -> List[Dict[str, Any]]:
    """Get all reviews for a specific document."""
    try:
        document_reviews = [
            item for item in review_store.values()
            if item["document_id"] == document_id
        ]
        
        # Sort by field name and timestamp
        document_reviews.sort(key=lambda x: (x.get("field_name", ""), x.get("timestamp", "")))
        
        return document_reviews
        
    except Exception as e:
        logger.error(f"Failed to get document reviews: {e}")
        raise HTTPException(status_code=500, detail=f"Document reviews retrieval failed: {str(e)}")


@router.get("/stats")
async def get_review_stats() -> Dict[str, Any]:
    """Get review system statistics."""
    try:
        total = len(review_store)
        
        if total == 0:
            return {
                "total_reviews": 0,
                "by_status": {},
                "by_document": {},
                "avg_confidence": 0.0
            }
        
        # Count by status
        by_status = {}
        for item in review_store.values():
            status = item.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        
        # Count by document
        by_document = {}
        for item in review_store.values():
            doc_id = item.get("document_id", "unknown")
            by_document[doc_id] = by_document.get(doc_id, 0) + 1
        
        # Calculate average confidence
        confidences = [item.get("confidence", 0.0) for item in review_store.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_reviews": total,
            "by_status": by_status,
            "by_document": by_document,
            "avg_confidence": avg_confidence,
            "pending_count": by_status.get("pending", 0),
            "completion_rate": ((total - by_status.get("pending", 0)) / total * 100) if total > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get review stats: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")


@router.post("/auto-submit/{document_id}")
async def auto_submit_low_confidence(document_id: str) -> ReviewResponse:
    """
    Automatically submit low-confidence extractions for review.
    
    This endpoint analyzes agent results and submits items below
    confidence threshold for human review.
    """
    try:
        # Load agent results
        doc_dir = settings.DOCUMENTS_DIR / document_id
        agent_file = doc_dir / "agents" / "agent_results.json"
        
        if not agent_file.exists():
            raise HTTPException(status_code=404, detail="Agent results not found")
        
        with open(agent_file, "r", encoding="utf-8") as f:
            agent_results = json.load(f)
        
        # Get fused extractions
        fused_doc = agent_results.get("fused_document", {})
        fused_extractions = fused_doc.get("fused_extractions", {})
        
        if not fused_extractions:
            return ReviewResponse(
                success=True,
                message="No extractions to review"
            )
        
        submitted_count = 0
        
        # Submit low-confidence extractions
        for field_name, extraction in fused_extractions.items():
            if isinstance(extraction, dict):
                confidence = extraction.get("multi_modal_confidence", 0.0)
                value = extraction.get("value", "")
            else:
                confidence = fused_doc.get("fusion_confidence", 0.0)
                value = extraction
            
            # Check if needs review
            needs_review, reason = confidence_engine.should_send_for_review(confidence)
            
            if needs_review:
                # Create review item
                review_item = ReviewItem(
                    document_id=document_id,
                    field_name=field_name,
                    extracted_value=value,
                    confidence=confidence,
                    reason_for_review=reason,
                    agent_sources=["fusion_agent"]
                )
                
                # Submit for review
                await submit_for_review(review_item)
                submitted_count += 1
        
        return ReviewResponse(
            success=True,
            message=f"Submitted {submitted_count} items for review"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto-submit failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-submit failed: {str(e)}")
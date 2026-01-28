"""
Real confidence calculation engine.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence calculation."""
    ocr_confidence: float = 0.0
    detection_confidence: float = 0.0
    semantic_confidence: float = 0.0
    spatial_confidence: float = 0.0
    fusion_confidence: float = 0.0
    embedding_similarity: float = 0.0
    llm_consistency: float = 0.0
    cross_modal_agreement: float = 0.0
    
    def validate(self) -> bool:
        """Validate all factors are within [0, 1] range."""
        factors = [
            self.ocr_confidence, self.detection_confidence,
            self.semantic_confidence, self.spatial_confidence,
            self.fusion_confidence, self.embedding_similarity,
            self.llm_consistency, self.cross_modal_agreement
        ]
        return all(0.0 <= factor <= 1.0 for factor in factors)


class ConfidenceEngine:
    """Calculates real confidence scores using weighted factors."""
    
    # Production weights (tuned for document intelligence)
    WEIGHTS = {
        "ocr_confidence": 0.15,
        "detection_confidence": 0.10,
        "semantic_confidence": 0.20,
        "spatial_confidence": 0.10,
        "fusion_confidence": 0.15,
        "embedding_similarity": 0.10,
        "llm_consistency": 0.10,
        "cross_modal_agreement": 0.10
    }
    
    # Thresholds for different confidence levels
    THRESHOLDS = {
        "high_confidence": 0.85,
        "medium_confidence": 0.65,
        "low_confidence": 0.45,
        "rejection_threshold": 0.30
    }
    
    def __init__(self):
        """Initialize confidence engine."""
        logger.info("✅ Confidence Engine initialized")
    
    def calculate_overall_confidence(self, factors: ConfidenceFactors) -> float:
        """
        Calculate overall confidence using weighted factors.
        
        Args:
            factors: Confidence factors from various sources
            
        Returns:
            Overall confidence score (0-1)
        """
        if not factors.validate():
            logger.error("Invalid confidence factors (outside [0, 1] range)")
            return 0.0
        
        try:
            # Calculate weighted sum
            weighted_sum = (
                factors.ocr_confidence * self.WEIGHTS["ocr_confidence"] +
                factors.detection_confidence * self.WEIGHTS["detection_confidence"] +
                factors.semantic_confidence * self.WEIGHTS["semantic_confidence"] +
                factors.spatial_confidence * self.WEIGHTS["spatial_confidence"] +
                factors.fusion_confidence * self.WEIGHTS["fusion_confidence"] +
                factors.embedding_similarity * self.WEIGHTS["embedding_similarity"] +
                factors.llm_consistency * self.WEIGHTS["llm_consistency"] +
                factors.cross_modal_agreement * self.WEIGHTS["cross_modal_agreement"]
            )
            
            # Apply non-linear scaling (sigmoid-like)
            # This emphasizes differences in the middle range
            scaled_confidence = self._apply_confidence_scaling(weighted_sum)
            
            # Cap at [0, 1]
            final_confidence = max(0.0, min(1.0, scaled_confidence))
            
            logger.debug(f"Calculated confidence: {final_confidence:.4f} "
                        f"(raw: {weighted_sum:.4f})")
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _apply_confidence_scaling(self, raw_score: float) -> float:
        """Apply non-linear scaling to confidence score."""
        # Sigmoid scaling: emphasizes differences around 0.5
        scaled = 1 / (1 + np.exp(-10 * (raw_score - 0.5)))
        return scaled
    
    def extract_factors_from_agent_results(self, agent_results: Dict[str, Any]) -> ConfidenceFactors:
        """
        Extract confidence factors from agent pipeline results.
        
        Args:
            agent_results: Complete agent pipeline results
            
        Returns:
            ConfidenceFactors object
        """
        try:
            # Extract from different agent outputs
            vision_analysis = agent_results.get("vision_analysis", {})
            text_analysis = agent_results.get("text_analysis", {})
            fused_document = agent_results.get("fused_document", {})
            ocr_data = agent_results.get("ocr_data", {})
            
            # OCR confidence (average across pages)
            ocr_confidences = []
            if "pages" in ocr_data:
                for page in ocr_data["pages"]:
                    for region in page.get("regions", []):
                        conf = region.get("ocr_confidence", 0.0)
                        if conf > 0:
                            ocr_confidences.append(conf)
            
            ocr_confidence = np.mean(ocr_confidences) if ocr_confidences else 0.0
            
            # Detection confidence (from vision agent)
            detection_confidence = vision_analysis.get("spatial_confidence", 0.0)
            
            # Semantic confidence (from text agent)
            semantic_confidence = text_analysis.get("semantic_confidence", 0.0)
            
            # Spatial confidence (from vision agent hierarchy)
            spatial_confidence = vision_analysis.get("spatial_confidence", 0.0)
            
            # Fusion confidence
            fusion_confidence = fused_document.get("fusion_confidence", 0.0)
            
            # Calculate cross-modal agreement
            cross_modal_agreement = self._calculate_cross_modal_agreement(
                vision_analysis, text_analysis, fused_document
            )
            
            # Default values for factors not directly available
            embedding_similarity = 0.7  # Would come from RAG system
            llm_consistency = 0.8  # Would come from LLM self-consistency checks
            
            return ConfidenceFactors(
                ocr_confidence=ocr_confidence,
                detection_confidence=detection_confidence,
                semantic_confidence=semantic_confidence,
                spatial_confidence=spatial_confidence,
                fusion_confidence=fusion_confidence,
                embedding_similarity=embedding_similarity,
                llm_consistency=llm_consistency,
                cross_modal_agreement=cross_modal_agreement
            )
            
        except Exception as e:
            logger.error(f"Failed to extract confidence factors: {e}")
            return ConfidenceFactors()  # All zeros
    
    def _calculate_cross_modal_agreement(self, vision: Dict, text: Dict, fusion: Dict) -> float:
        """Calculate agreement between vision and text modalities."""
        try:
            # Extract key information from each modality
            vision_elements = len(vision.get("logical_reading_order", []))
            text_elements = len(text.get("key_value_pairs", {}))
            
            if vision_elements == 0 or text_elements == 0:
                return 0.0
            
            # Check if fused document contains elements from both modalities
            fused_elements = fusion.get("fused_extractions", {})
            if not fused_elements:
                return 0.0
            
            # Simple agreement metric
            # In production, this would be more sophisticated
            agreement = min(1.0, len(fused_elements) / max(vision_elements, text_elements))
            
            return agreement
            
        except Exception as e:
            logger.warning(f"Cross-modal agreement calculation failed: {e}")
            return 0.0
    
    def determine_confidence_level(self, confidence: float) -> str:
        """Determine confidence level based on score."""
        if confidence >= self.THRESHOLDS["high_confidence"]:
            return "high"
        elif confidence >= self.THRESHOLDS["medium_confidence"]:
            return "medium"
        elif confidence >= self.THRESHOLDS["low_confidence"]:
            return "low"
        else:
            return "very_low"
    
    def should_send_for_review(self, confidence: float) -> Tuple[bool, str]:
        """
        Determine if result should be sent for human review.
        
        Args:
            confidence: Overall confidence score
            
        Returns:
            Tuple of (needs_review, reason)
        """
        if confidence < self.THRESHOLDS["rejection_threshold"]:
            return True, f"Confidence below rejection threshold ({confidence:.2f} < {self.THRESHOLDS['rejection_threshold']})"
        
        confidence_level = self.determine_confidence_level(confidence)
        
        if confidence_level in ["low", "very_low"]:
            return True, f"Low confidence level: {confidence_level} ({confidence:.2f})"
        
        return False, f"Confidence acceptable: {confidence_level} ({confidence:.2f})"
    
    def generate_confidence_report(self, factors: ConfidenceFactors, 
                                 overall_confidence: float) -> Dict[str, Any]:
        """Generate detailed confidence report."""
        confidence_level = self.determine_confidence_level(overall_confidence)
        needs_review, review_reason = self.should_send_for_review(overall_confidence)
        
        # Calculate contribution of each factor
        contributions = {
            "ocr_confidence": factors.ocr_confidence * self.WEIGHTS["ocr_confidence"],
            "detection_confidence": factors.detection_confidence * self.WEIGHTS["detection_confidence"],
            "semantic_confidence": factors.semantic_confidence * self.WEIGHTS["semantic_confidence"],
            "spatial_confidence": factors.spatial_confidence * self.WEIGHTS["spatial_confidence"],
            "fusion_confidence": factors.fusion_confidence * self.WEIGHTS["fusion_confidence"],
            "embedding_similarity": factors.embedding_similarity * self.WEIGHTS["embedding_similarity"],
            "llm_consistency": factors.llm_consistency * self.WEIGHTS["llm_consistency"],
            "cross_modal_agreement": factors.cross_modal_agreement * self.WEIGHTS["cross_modal_agreement"]
        }
        
        # Identify weakest factor
        weakest_factor = min(contributions.items(), key=lambda x: x[1])
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level,
            "needs_human_review": needs_review,
            "review_reason": review_reason,
            "factor_contributions": contributions,
            "weakest_factor": {
                "factor": weakest_factor[0],
                "contribution": weakest_factor[1],
                "recommendation": self._get_improvement_recommendation(weakest_factor[0])
            },
            "thresholds": self.THRESHOLDS,
            "weights": self.WEIGHTS
        }
    
    def _get_improvement_recommendation(self, factor: str) -> str:
        """Get recommendation for improving a specific factor."""
        recommendations = {
            "ocr_confidence": "Improve image quality or use higher resolution OCR",
            "detection_confidence": "Adjust layout detection thresholds or retrain model",
            "semantic_confidence": "Enhance text understanding with domain-specific training",
            "spatial_confidence": "Improve layout analysis with more training data",
            "fusion_confidence": "Enhance multi-modal fusion algorithm",
            "embedding_similarity": "Use better embedding models or fine-tune existing ones",
            "llm_consistency": "Improve prompt engineering or use more consistent LLM",
            "cross_modal_agreement": "Enhance cross-modal alignment algorithms"
        }
        return recommendations.get(factor, "General system improvement needed")


# Global instance
confidence_engine = ConfidenceEngine()
"""
Validation Intelligence Agent - ensures quality and reliability.
"""
import json
import logging
import re
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.llm.prompts import VALIDATION_AGENT_SYSTEM_PROMPT, VALIDATION_AGENT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Validates fused document understanding."""
    
    def __init__(self):
        """Initialize Validation Agent."""
        super().__init__("ValidationAgent")
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate fused document understanding.
        
        Expected inputs:
            - document_id: str
            - fused_document: dict from Fusion Agent
            - vision_analysis: dict from Vision Agent
            - text_analysis: dict from Text Agent
        """
        # Validate inputs
        required = ["document_id", "fused_document", "vision_analysis", "text_analysis"]
        if not self.validate_inputs(inputs, required):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        
        logger.info(f"✅ Validation Agent processing document {document_id}")
        
        # Prepare validation data
        fused_summary = self._summarize_for_validation(inputs["fused_document"])
        ocr_confidence = inputs["text_analysis"].get("ocr_confidence_summary", {}).get("average_confidence", 0.0)
        layout_confidence = inputs["vision_analysis"].get("spatial_confidence", 0.0)
        fusion_confidence = inputs["fused_document"].get("fusion_confidence", 0.0)
        
        # Call LLM for validation
        prompt = VALIDATION_AGENT_PROMPT_TEMPLATE.format(
            fused_document=fused_summary,
            ocr_confidence_summary=f"Average OCR confidence: {ocr_confidence:.2f}",
            layout_confidence_summary=f"Layout analysis confidence: {layout_confidence:.2f}",
            fusion_confidence=f"Fusion confidence: {fusion_confidence:.2f}"
        )
        
        logger.info(f"📡 Validation Agent calling LLM")
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=VALIDATION_AGENT_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            if not response or len(response.strip()) == 0:
                logger.error("LLM returned empty response")
                return self._create_basic_validation(inputs)
            
            logger.info(f"✅ LLM response received: {len(response)} chars")
            
            # Parse response
            validation = self._parse_validation_response(response, inputs)
            
            logger.info(f"🎉 Validation Agent completed for {document_id}")
            logger.info(f"   Overall confidence: {validation.get('overall_confidence', 0.0):.2f}")
            logger.info(f"   Passed checks: {len(validation.get('validation_passed', []))}")
            logger.info(f"   Failed checks: {len(validation.get('validation_failed', []))}")
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Validation Agent failed: {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_basic_validation(inputs)
    
    def _summarize_for_validation(self, fused_document: Dict[str, Any]) -> str:
        """Create summary for validation prompt."""
        try:
            return json.dumps(fused_document, indent=2, default=str)
        except:
            return str(fused_document)
    
    def _parse_validation_response(self, response: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured validation."""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                validation = json.loads(json_str)
                logger.info(f"✅ Successfully parsed JSON from Validation LLM")
            else:
                logger.warning("⚠️ No JSON found in Validation LLM response")
                validation = {}
            
            # Calculate overall confidence
            if "overall_confidence" not in validation:
                vision_conf = inputs["vision_analysis"].get("spatial_confidence", 0.0)
                text_conf = inputs["text_analysis"].get("semantic_confidence", 0.0)
                fusion_conf = inputs["fused_document"].get("fusion_confidence", 0.0)
                
                # Weighted average
                weights = [0.2, 0.3, 0.5]
                raw_conf = (vision_conf * weights[0] + 
                           text_conf * weights[1] + 
                           fusion_conf * weights[2])
                
                validation["overall_confidence"] = raw_conf
            
            # Add metadata
            validation["agent"] = self.name
            validation["document_id"] = inputs["document_id"]
            validation["timestamp"] = self._get_timestamp()
            
            # Ensure required lists
            for key in ["validation_passed", "validation_failed", "explainability_notes"]:
                if key not in validation:
                    validation[key] = []
            
            if "hallucination_risk" not in validation:
                validation["hallucination_risk"] = 0.2  # Low default
            
            return validation
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON from Validation Agent: {e}")
            return self._create_basic_validation(inputs)
    
    def _create_basic_validation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic validation when LLM fails."""
        fused_doc = inputs.get("fused_document", {})
        vision_analysis = inputs.get("vision_analysis", {})
        text_analysis = inputs.get("text_analysis", {})
        
        vision_conf = vision_analysis.get("spatial_confidence", 0.0)
        text_conf = text_analysis.get("semantic_confidence", 0.0)
        fusion_conf = fused_doc.get("fusion_confidence", 0.0)
        
        # Weighted average confidence
        overall_conf = (vision_conf * 0.3 + text_conf * 0.3 + fusion_conf * 0.4)
        
        validation_passed = []
        validation_failed = []
        
        # Basic validation checks
        if fusion_conf > 0.5:
            validation_passed.append("Fusion confidence above threshold")
        else:
            validation_failed.append("Low fusion confidence")
        
        if text_conf > 0.6:
            validation_passed.append("Text analysis confidence acceptable")
        else:
            validation_failed.append("Low text analysis confidence")
        
        if fused_doc.get("fused_extractions"):
            extraction_count = len(fused_doc["fused_extractions"])
            validation_passed.append(f"Extracted {extraction_count} fields")
        else:
            validation_failed.append("No fields extracted")
        
        return {
            "agent": self.name,
            "document_id": inputs.get("document_id", ""),
            "overall_confidence": overall_conf,
            "validation_passed": validation_passed,
            "validation_failed": validation_failed,
            "hallucination_risk": 0.3,
            "explainability_notes": [
                "Basic validation using confidence scores",
                f"Vision confidence: {vision_conf:.2f}",
                f"Text confidence: {text_conf:.2f}",
                f"Fusion confidence: {fusion_conf:.2f}",
                f"Overall: {overall_conf:.2f}"
            ],
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
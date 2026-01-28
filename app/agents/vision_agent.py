"""
Vision Intelligence Agent - analyzes document layouts.
"""
import json
import logging
import re
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.llm.prompts import VISION_AGENT_SYSTEM_PROMPT, VISION_AGENT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class VisionAgent(BaseAgent):
    """Analyzes document layouts with spatial reasoning."""
    
    def __init__(self):
        """Initialize Vision Agent."""
        super().__init__("VisionAgent")
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze document layout.
        
        Expected inputs:
            - document_id: str
            - layout_data: dict from Phase 2
        """
        # Validate inputs
        if not self.validate_inputs(inputs, ["document_id", "layout_data"]):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        layout_data = inputs["layout_data"]
        
        logger.info(f"👁️ Vision Agent processing document {document_id}")
        
        # Prepare layout summary
        pages_summary = self._summarize_layout(layout_data)
        
        # Call LLM for spatial analysis
        prompt = VISION_AGENT_PROMPT_TEMPLATE.format(pages_summary=pages_summary)
        
        logger.info(f"📡 Vision Agent calling LLM")
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=VISION_AGENT_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            if not response or len(response.strip()) == 0:
                logger.error("LLM returned empty response")
                return self._create_basic_analysis(layout_data, document_id)
            
            logger.info(f"✅ LLM response received: {len(response)} chars")
            
            # Parse response
            analysis = self._parse_vision_response(response, layout_data, document_id)
            
            logger.info(f"🎉 Vision Agent completed for {document_id}")
            logger.info(f"   Spatial confidence: {analysis.get('spatial_confidence', 0.0):.2f}")
            logger.info(f"   Reading order regions: {len(analysis.get('logical_reading_order', []))}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Vision Agent failed: {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_basic_analysis(layout_data, document_id)
    
    def _summarize_layout(self, layout_data: Dict[str, Any]) -> str:
        """Create text summary of layout for LLM."""
        summary_parts = []
        
        if "pages" in layout_data:
            for page in layout_data["pages"]:
                page_num = page.get("page_number", 1)
                detections = page.get("detections", [])
                
                page_summary = f"Page {page_num}: {len(detections)} regions"
                for i, det in enumerate(detections):
                    region_id = f"p{page_num}_r{i}"
                    bbox = det.get("bbox", {})
                    confidence = det.get("confidence", 0.0)
                    page_summary += f"\n  {region_id}: {det.get('type', 'unknown')} at ({bbox.get('x1', 0):.2f}, {bbox.get('y1', 0):.2f}) - confidence: {confidence:.2f}"
                
                summary_parts.append(page_summary)
        
        return "\n\n".join(summary_parts)
    
    def _parse_vision_response(self, response: str, layout_data: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis."""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
                logger.info(f"✅ Successfully parsed JSON from Vision LLM")
            else:
                logger.warning("⚠️ No JSON found in Vision LLM response")
                analysis = {}
            
            # Ensure required fields
            if "logical_reading_order" not in analysis:
                analysis["logical_reading_order"] = self._infer_reading_order(layout_data)
            
            if "spatial_confidence" not in analysis:
                analysis["spatial_confidence"] = 0.7  # Default moderate confidence
            
            if "hierarchy_analysis" not in analysis:
                analysis["hierarchy_analysis"] = {}
            
            if "region_importance_scores" not in analysis:
                analysis["region_importance_scores"] = {}
            
            if "visual_validation_notes" not in analysis:
                analysis["visual_validation_notes"] = ["Layout analysis completed"]
            
            analysis["agent"] = self.name
            analysis["document_id"] = document_id
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON from Vision Agent: {e}")
            return self._create_basic_analysis(layout_data, document_id)
    
    def _infer_reading_order(self, layout_data: Dict[str, Any]) -> List[str]:
        """Infer reading order based on positions."""
        reading_order = []
        
        if "pages" in layout_data:
            for page in layout_data["pages"]:
                page_num = page.get("page_number", 1)
                detections = page.get("detections", [])
                
                # Sort by Y then X (top-left to bottom-right)
                sorted_detections = sorted(
                    detections,
                    key=lambda d: (d.get("bbox", {}).get("y1", 0), d.get("bbox", {}).get("x1", 0))
                )
                
                for i, det in enumerate(sorted_detections):
                    region_id = f"p{page_num}_r{i}"
                    reading_order.append(region_id)
        
        return reading_order
    
    def _create_basic_analysis(self, layout_data: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        """Create basic analysis when LLM fails."""
        return {
            "agent": self.name,
            "document_id": document_id,
            "logical_reading_order": self._infer_reading_order(layout_data),
            "hierarchy_analysis": {},
            "region_importance_scores": {},
            "visual_validation_notes": ["Basic layout analysis - LLM unavailable"],
            "spatial_confidence": 0.5
        }
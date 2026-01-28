"""
Fusion Intelligence Agent - combines vision and text analysis.
"""
import json
import logging
import re
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.llm.prompts import FUSION_AGENT_SYSTEM_PROMPT, FUSION_AGENT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class FusionAgent(BaseAgent):
    """Fuses vision and text analysis into unified understanding."""
    
    def __init__(self):
        """Initialize Fusion Agent."""
        super().__init__("FusionAgent")
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse vision and text analysis.
        
        Expected inputs:
            - document_id: str
            - vision_analysis: dict from Vision Agent
            - text_analysis: dict from Text Agent
            - layout_data: dict from Phase 2
            - ocr_data: dict from Phase 3
        """
        # Validate inputs
        required = ["document_id", "vision_analysis", "text_analysis"]
        if not self.validate_inputs(inputs, required):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        
        logger.info(f"🤝 Fusion Agent processing document {document_id}")
        
        # Prepare analysis summaries
        vision_summary = self._summarize_for_fusion(inputs["vision_analysis"])
        text_summary = self._summarize_for_fusion(inputs["text_analysis"])
        
        # Call LLM for fusion
        prompt = FUSION_AGENT_PROMPT_TEMPLATE.format(
            vision_analysis=vision_summary,
            text_analysis=text_summary
        )
        
        logger.info(f"📡 Fusion Agent calling LLM")
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=FUSION_AGENT_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            if not response or len(response.strip()) == 0:
                logger.error("LLM returned empty response")
                return self._create_basic_fusion(inputs)
            
            logger.info(f"✅ LLM response received: {len(response)} chars")
            
            # Parse response
            fusion = self._parse_fusion_response(response, inputs)
            
            logger.info(f"🎉 Fusion Agent completed for {document_id}")
            logger.info(f"   Fusion confidence: {fusion.get('fusion_confidence', 0.0):.2f}")
            if fusion.get("fused_extractions"):
                logger.info(f"   Fused extractions: {len(fusion['fused_extractions'])}")
            
            return fusion
            
        except Exception as e:
            logger.error(f"❌ Fusion Agent failed: {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_basic_fusion(inputs)
    
    def _summarize_for_fusion(self, analysis: Dict[str, Any]) -> str:
        """Create summary for fusion prompt."""
        try:
            return json.dumps(analysis, indent=2, default=str)
        except:
            return str(analysis)
    
    def _parse_fusion_response(self, response: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured fusion."""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                fusion = json.loads(json_str)
                logger.info(f"✅ Successfully parsed JSON from Fusion LLM")
            else:
                logger.warning("⚠️ No JSON found in Fusion LLM response")
                fusion = {}
            
            # Add required fields
            fusion["traceability"] = {
                "vision_confidence": inputs["vision_analysis"].get("spatial_confidence", 0.0),
                "text_confidence": inputs["text_analysis"].get("semantic_confidence", 0.0),
                "ocr_confidence": inputs["text_analysis"].get("ocr_confidence_summary", {}).get("average_confidence", 0.0),
                "agent_versions": {
                    "vision": inputs["vision_analysis"].get("agent", "unknown"),
                    "text": inputs["text_analysis"].get("agent", "unknown"),
                    "fusion": self.name
                }
            }
            
            # Calculate multi-modal confidence
            if "fusion_confidence" not in fusion:
                vision_conf = inputs["vision_analysis"].get("spatial_confidence", 0.5)
                text_conf = inputs["text_analysis"].get("semantic_confidence", 0.5)
                fusion["fusion_confidence"] = (vision_conf * 0.4 + text_conf * 0.6)
                logger.info(f"📊 Calculated fusion confidence: {fusion['fusion_confidence']:.2f}")
            
            if "unified_structure" not in fusion:
                fusion["unified_structure"] = {
                    "document_type": inputs["text_analysis"].get("document_type", "unknown"),
                    "pages": []
                }
            
            # In the section where you create fused_extractions:
            if "fused_extractions" not in fusion:
    # Use text analysis extractions as baseline
                text_extractions = inputs["text_analysis"].get("key_value_pairs", {})
                key_entities = inputs["text_analysis"].get("key_entities", {})
    
                fused_extractions = {}
    
    # First try key_value_pairs
                for key, value in text_extractions.items():
                    fused_extractions[key] = {
                        "value": value,  # ACTUAL VALUE, not nested dict
                        "multi_modal_confidence": text_conf,
                        "source": "text_agent"
                }
    
    # Also add entities
                for entity_type, entities in key_entities.items():
                    if entities and isinstance(entities, list):
                        fused_extractions[entity_type] = {
                            "value": entities[0],  # ACTUAL VALUE
                            "multi_modal_confidence": text_conf,
                            "source": "text_agent_entities"
                        }
    
                fusion["fused_extractions"] = fused_extractions
            
            if "conflict_resolutions" not in fusion:
                fusion["conflict_resolutions"] = []
            
            if "alignment_notes" not in fusion:
                fusion["alignment_notes"] = ["Vision and text analysis aligned"]
            
            fusion["agent"] = self.name
            fusion["document_id"] = inputs["document_id"]
            
            # Log results
            if fusion.get("fused_extractions"):
                logger.info(f"📊 Fused extractions:")
                for key, info in list(fusion["fused_extractions"].items())[:5]:
                    value = info.get("value", info) if isinstance(info, dict) else info
                    logger.info(f"   • {key}: {value}")
            
            return fusion
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON from Fusion Agent: {e}")
            return self._create_basic_fusion(inputs)
    
    def _create_basic_fusion(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic fusion when LLM fails."""
        vision_analysis = inputs.get("vision_analysis", {})
        text_analysis = inputs.get("text_analysis", {})
        
        vision_conf = vision_analysis.get("spatial_confidence", 0.5)
        text_conf = text_analysis.get("semantic_confidence", 0.5)
        fusion_conf = (vision_conf * 0.4 + text_conf * 0.6)
        
        # Create basic fused extractions from text analysis
        text_extractions = text_analysis.get("key_value_pairs", {})
        fused_extractions = {}
        
        for key, value in text_extractions.items():
            if isinstance(value, dict):
                fused_extractions[key] = {
                    "value": value.get("value", value),
                    "multi_modal_confidence": text_conf,
                    "source": "text_agent"
                }
            else:
                fused_extractions[key] = {
                    "value": value,
                    "multi_modal_confidence": text_conf,
                    "source": "text_agent"
                }
        
        return {
            "agent": self.name,
            "document_id": inputs.get("document_id", ""),
            "unified_structure": {
                "document_type": text_analysis.get("document_type", "unknown"),
                "pages": []
            },
            "fused_extractions": fused_extractions,
            "conflict_resolutions": [],
            "fusion_confidence": fusion_conf,
            "alignment_notes": ["Basic fusion - using text analysis data"],
            "traceability": {
                "vision_confidence": vision_conf,
                "text_confidence": text_conf,
                "ocr_confidence": text_analysis.get("ocr_confidence_summary", {}).get("average_confidence", 0.0),
                "agent_versions": {
                    "vision": vision_analysis.get("agent", "unknown"),
                    "text": text_analysis.get("agent", "unknown"),
                    "fusion": self.name
                }
            }
        }
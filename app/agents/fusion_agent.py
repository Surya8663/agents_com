"""
Fusion Intelligence Agent - EXTENDED with RAG context.
"""
import json
import logging
import re
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.llm.prompts import FUSION_AGENT_SYSTEM_PROMPT, FUSION_AGENT_PROMPT_TEMPLATE
from app.rag.retriever import retriever
from app.rag.multimodal_schema import RAGQuery

logger = logging.getLogger(__name__)


class FusionAgent(BaseAgent):
    """Fuses vision and text analysis with RAG context - EXTENDED."""
    
    def __init__(self):
        """Initialize Fusion Agent with RAG capabilities."""
        super().__init__("FusionAgent")
        self.retriever = retriever
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse vision and text analysis with RAG context.
        
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
        
        logger.info(f"🤝 Fusion Agent processing document {document_id} with RAG context")
        
        # Prepare analysis summaries
        vision_summary = self._summarize_for_fusion(inputs["vision_analysis"])
        text_summary = self._summarize_for_fusion(inputs["text_analysis"])
        
        # Retrieve RAG context for document understanding
        rag_context = await self._retrieve_rag_context(document_id)
        
        # Call LLM for fusion with RAG context
        prompt = self._build_fusion_prompt(vision_summary, text_summary, rag_context)
        
        logger.info(f"📡 Fusion Agent calling LLM with RAG context")
        
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
            fusion = self._parse_fusion_response(response, inputs, rag_context)
            
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
    
    async def _retrieve_rag_context(self, document_id: str) -> str:
        """Retrieve RAG context for document understanding."""
        try:
            # Query for document structure and key information
            queries = [
                "What is the overall document structure?",
                "What are the key sections or headings?",
                "What type of document is this?",
                "What are the most important data points?"
            ]
            
            contexts = []
            for query in queries:
                rag_query = RAGQuery(
                    query=query,
                    document_id=document_id,
                    top_k=3,
                    similarity_threshold=0.7
                )
                
                response = await self.retriever.query(rag_query)
                if response.answer and response.confidence > 0.6:
                    contexts.append(f"Q: {query}\nA: {response.answer} (confidence: {response.confidence:.2f})")
            
            if contexts:
                rag_context = "RAG Context from similar documents:\n" + "\n\n".join(contexts)
                logger.info(f"Retrieved RAG context with {len(contexts)} insights")
                return rag_context
            
            return "No RAG context available for this document."
            
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
            return "RAG context unavailable."
    
    def _build_fusion_prompt(self, vision_summary: str, text_summary: str, rag_context: str) -> str:
        """Build fusion prompt with RAG context."""
        prompt = FUSION_AGENT_PROMPT_TEMPLATE.format(
            vision_analysis=vision_summary,
            text_analysis=text_summary
        )
        
        # Add RAG context section
        prompt += f"\n\nADDITIONAL CONTEXT FROM SIMILAR DOCUMENTS:\n{rag_context}\n\n"
        prompt += "Use this additional context to improve your fusion analysis, especially for:\n"
        prompt += "1. Document type identification\n2. Key field extraction\n3. Structure validation\n"
        prompt += "4. Confidence calibration\n"
        
        return prompt
    
    def _parse_fusion_response(self, response: str, inputs: Dict[str, Any], 
                              rag_context: str) -> Dict[str, Any]:
        """Parse LLM response with RAG context tracking."""
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
            
            # Add RAG context metadata
            fusion["rag_context_used"] = rag_context != "No RAG context available for this document."
            fusion["rag_insights_count"] = rag_context.count("Q:") if rag_context else 0
            
            # Add required fields
            fusion["traceability"] = {
                "vision_confidence": inputs["vision_analysis"].get("spatial_confidence", 0.0),
                "text_confidence": inputs["text_analysis"].get("semantic_confidence", 0.0),
                "ocr_confidence": inputs["text_analysis"].get("ocr_confidence_summary", {}).get("average_confidence", 0.0),
                "rag_context_used": fusion.get("rag_context_used", False),
                "agent_versions": {
                    "vision": inputs["vision_analysis"].get("agent", "unknown"),
                    "text": inputs["text_analysis"].get("agent", "unknown"),
                    "fusion": self.name
                }
            }
            
            # Calculate multi-modal confidence with RAG boost
            if "fusion_confidence" not in fusion:
                vision_conf = inputs["vision_analysis"].get("spatial_confidence", 0.5)
                text_conf = inputs["text_analysis"].get("semantic_confidence", 0.5)
                rag_boost = 0.1 if fusion.get("rag_context_used") else 0.0
                
                fusion["fusion_confidence"] = min(1.0, (vision_conf * 0.4 + text_conf * 0.6) + rag_boost)
                logger.info(f"📊 Calculated fusion confidence: {fusion['fusion_confidence']:.2f} (RAG boost: {rag_boost})")
            
            # Enhanced unified structure with RAG insights
            if "unified_structure" not in fusion:
                fusion["unified_structure"] = {
                    "document_type": inputs["text_analysis"].get("document_type", "unknown"),
                    "pages": [],
                    "rag_informed": fusion.get("rag_context_used", False)
                }
            
            # Enhanced fused extractions with source tracking
            if "fused_extractions" not in fusion:
                text_extractions = inputs["text_analysis"].get("key_value_pairs", {})
                key_entities = inputs["text_analysis"].get("key_entities", {})
                text_conf = inputs["text_analysis"].get("semantic_confidence", 0.5)
                
                fused_extractions = {}
                
                # Add key-value pairs with enhanced metadata
                for key, value in text_extractions.items():
                    fused_extractions[key] = {
                        "value": value,
                        "multi_modal_confidence": text_conf,
                        "source": "text_agent",
                        "has_rag_context": fusion.get("rag_context_used", False),
                        "extraction_method": "semantic_analysis"
                    }
                
                # Add entities with enhanced metadata
                for entity_type, entities in key_entities.items():
                    if entities and isinstance(entities, list):
                        fused_extractions[entity_type] = {
                            "value": entities[0],
                            "multi_modal_confidence": text_conf,
                            "source": "text_agent_entities",
                            "has_rag_context": fusion.get("rag_context_used", False),
                            "extraction_method": "entity_recognition"
                        }
                
                fusion["fused_extractions"] = fused_extractions
            
            # Enhanced conflict resolutions
            if "conflict_resolutions" not in fusion:
                fusion["conflict_resolutions"] = []
            
            # Enhanced alignment notes with RAG insights
            if "alignment_notes" not in fusion:
                if fusion.get("rag_context_used"):
                    fusion["alignment_notes"] = [
                        "Vision and text analysis aligned",
                        "RAG context provided additional validation"
                    ]
                else:
                    fusion["alignment_notes"] = ["Vision and text analysis aligned"]
            
            fusion["agent"] = self.name
            fusion["document_id"] = inputs["document_id"]
            fusion["version"] = "2.0_with_rag"  # Version tracking
            
            # Log results with RAG info
            if fusion.get("fused_extractions"):
                logger.info(f"📊 Fused extractions (RAG enabled: {fusion.get('rag_context_used', False)}):")
                for key, info in list(fusion["fused_extractions"].items())[:5]:
                    value = info.get("value", info) if isinstance(info, dict) else info
                    confidence = info.get("multi_modal_confidence", 0.0) if isinstance(info, dict) else 0.0
                    logger.info(f"   • {key}: {value} (conf: {confidence:.2f})")
            
            return fusion
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON from Fusion Agent: {e}")
            return self._create_basic_fusion(inputs)
    
    def _create_basic_fusion(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic fusion when LLM fails - enhanced with version tracking."""
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
                    "source": "text_agent",
                    "has_rag_context": False,
                    "extraction_method": "fallback",
                    "version": "1.0_basic"
                }
            else:
                fused_extractions[key] = {
                    "value": value,
                    "multi_modal_confidence": text_conf,
                    "source": "text_agent",
                    "has_rag_context": False,
                    "extraction_method": "fallback",
                    "version": "1.0_basic"
                }
        
        return {
            "agent": self.name,
            "document_id": inputs.get("document_id", ""),
            "version": "1.0_basic",
            "unified_structure": {
                "document_type": text_analysis.get("document_type", "unknown"),
                "pages": [],
                "rag_informed": False
            },
            "fused_extractions": fused_extractions,
            "conflict_resolutions": [],
            "fusion_confidence": fusion_conf,
            "alignment_notes": ["Basic fusion - using text analysis data", "RAG context unavailable"],
            "rag_context_used": False,
            "traceability": {
                "vision_confidence": vision_conf,
                "text_confidence": text_conf,
                "ocr_confidence": text_analysis.get("ocr_confidence_summary", {}).get("average_confidence", 0.0),
                "rag_context_used": False,
                "agent_versions": {
                    "vision": vision_analysis.get("agent", "unknown"),
                    "text": text_analysis.get("agent", "unknown"),
                    "fusion": self.name
                }
            }
        }
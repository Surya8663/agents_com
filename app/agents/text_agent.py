"""
Text Intelligence Agent - extracts semantic meaning from OCR.
"""
import json
import logging
import re
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.llm.prompts import TEXT_AGENT_SYSTEM_PROMPT, TEXT_AGENT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class TextAgent(BaseAgent):
    """Extracts semantic information from OCR text."""
    
    def __init__(self):
        """Initialize Text Agent."""
        super().__init__("TextAgent")
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract semantic information from OCR.
        
        Expected inputs:
            - document_id: str
            - ocr_data: dict from Phase 3
        """
        # Validate inputs
        if not self.validate_inputs(inputs, ["document_id", "ocr_data"]):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        ocr_data = inputs["ocr_data"]
        
        logger.info(f"🔤 Text Agent processing document {document_id}")
        
        # Prepare OCR summary
        ocr_summary = self._summarize_ocr(ocr_data)
        
        # Calculate OCR confidence summary
        confidence_summary = self._calculate_confidence_summary(ocr_data)
        
        # Call LLM for semantic extraction
        prompt = TEXT_AGENT_PROMPT_TEMPLATE.format(ocr_summary=ocr_summary)
        
        logger.info(f"📡 Text Agent calling LLM with {len(ocr_summary)} chars summary")
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=TEXT_AGENT_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            if not response or len(response.strip()) == 0:
                logger.error("LLM returned empty response")
                return {
                    "agent": self.name,
                    "document_id": document_id,
                    "error": "LLM returned empty response",
                    "document_type": self._infer_document_type(ocr_data),
                    "semantic_confidence": confidence_summary["average_confidence"],
                    "ocr_confidence_summary": confidence_summary,
                    "key_entities": {},
                    "key_value_pairs": {},
                    "table_extractions": []
                }
            
            logger.info(f"✅ LLM response received: {len(response)} chars")
            
            # Parse response
            analysis = self._parse_text_response(response, ocr_data, confidence_summary)
            
            logger.info(f"🎉 Text Agent completed for {document_id}")
            logger.info(f"   Document type: {analysis.get('document_type', 'unknown')}")
            if analysis.get("key_value_pairs"):
                logger.info(f"   Extracted {len(analysis['key_value_pairs'])} key-value pairs")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Text Agent failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "agent": self.name,
                "document_id": document_id,
                "error": f"Text Agent failed: {str(e)}",
                "document_type": self._infer_document_type(ocr_data),
                "semantic_confidence": confidence_summary["average_confidence"],
                "ocr_confidence_summary": confidence_summary,
                "key_entities": {},
                "key_value_pairs": {},
                "table_extractions": []
            }
    
    def _summarize_ocr(self, ocr_data: Dict[str, Any]) -> str:
        """Create text summary of OCR for LLM."""
        summary_parts = []
        
        if "pages" in ocr_data:
            for page in ocr_data["pages"]:
                page_num = page.get("page_number", 1)
                regions = page.get("regions", [])
                
                page_summary = f"Page {page_num}: {len(regions)} OCR regions"
                for i, region in enumerate(regions):
                    region_id = f"p{page_num}_r{i}"
                    text = region.get("ocr_text", "")
                    confidence = region.get("ocr_confidence", 0.0)
                    
                    if text and len(text.strip()) > 0:
                        preview = text[:100] + "..." if len(text) > 100 else text
                        page_summary += f"\n  {region_id} ({confidence:.2f}): {preview}"
                
                summary_parts.append(page_summary)
        
        return "\n\n".join(summary_parts)
    
    def _calculate_confidence_summary(self, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate OCR confidence statistics."""
        confidences = []
        total_regions = 0
        text_regions = 0
        
        if "pages" in ocr_data:
            for page in ocr_data["pages"]:
                regions = page.get("regions", [])
                total_regions += len(regions)
                
                for region in regions:
                    confidence = region.get("ocr_confidence", 0.0)
                    text = region.get("ocr_text", "")
                    
                    if confidence > 0:
                        confidences.append(confidence)
                    
                    if text and len(text.strip()) > 0:
                        text_regions += 1
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "average_confidence": avg_confidence,
            "total_regions": total_regions,
            "text_regions": text_regions,
            "confidence_distribution": {
                "high": len([c for c in confidences if c > 0.8]),
                "medium": len([c for c in confidences if 0.5 < c <= 0.8]),
                "low": len([c for c in confidences if c <= 0.5])
            }
        }
    
    def _parse_text_response(self, response: str, ocr_data: Dict[str, Any], 
                           confidence_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured extraction."""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
                logger.info(f"✅ Successfully parsed JSON from LLM response")
            else:
                logger.warning("⚠️ No JSON found in LLM response, using fallback parsing")
                analysis = {}
            
            # Add required fields
            analysis["ocr_confidence_summary"] = confidence_summary
            
            if "document_type" not in analysis:
                doc_type = self._infer_document_type(ocr_data)
                analysis["document_type"] = doc_type
                logger.info(f"📄 Inferred document type: {doc_type}")
            
            if "semantic_confidence" not in analysis:
                analysis["semantic_confidence"] = confidence_summary["average_confidence"]
            
            if "key_entities" not in analysis:
                analysis["key_entities"] = {}
            
            if "key_value_pairs" not in analysis:
                analysis["key_value_pairs"] = {}
            
            if "table_extractions" not in analysis:
                analysis["table_extractions"] = []
            
            analysis["agent"] = self.name
            
            # Log extraction results
            kv_pairs = analysis.get("key_value_pairs", {})
            if kv_pairs:
                logger.info(f"📊 Extracted key-value pairs:")
                for key, value in list(kv_pairs.items())[:5]:  # Show first 5
                    logger.info(f"   • {key}: {value}")
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON from Text Agent: {e}")
            return {
                "agent": self.name,
                "document_type": self._infer_document_type(ocr_data),
                "semantic_confidence": confidence_summary["average_confidence"],
                "ocr_confidence_summary": confidence_summary,
                "key_entities": {},
                "key_value_pairs": {},
                "table_extractions": [],
                "parsing_error": str(e),
                "raw_response_preview": response[:500] + "..." if len(response) > 500 else response
            }
    
    def _infer_document_type(self, ocr_data: Dict[str, Any]) -> str:
        """Infer document type from content (simple fallback)."""
        all_text = ""
        
        if "pages" in ocr_data:
            for page in ocr_data["pages"]:
                for region in page.get("regions", []):
                    text = region.get("ocr_text", "")
                    all_text += text + " "
        
        all_text_lower = all_text.lower()
        
        # Simple keyword matching
        if "invoice" in all_text_lower or "total" in all_text_lower or "amount" in all_text_lower:
            return "invoice"
        elif "contract" in all_text_lower or "agreement" in all_text_lower:
            return "contract"
        elif "report" in all_text_lower or "analysis" in all_text_lower:
            return "report"
        elif "dear" in all_text_lower or "sincerely" in all_text_lower:
            return "letter"
        elif "form" in all_text_lower or "application" in all_text_lower:
            return "form"
        
        return "unknown"
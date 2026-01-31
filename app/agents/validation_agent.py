"""
Validation Intelligence Agent - COMPLETE PRODUCTION FIX
"""
import json
import logging
import re
from datetime import datetime
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
        """Validate fused document understanding."""
        required = ["document_id", "fused_document", "vision_analysis", "text_analysis"]
        if not self.validate_inputs(inputs, required):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        logger.info(f"✅ Validation Agent processing document {document_id}")
        
        fused_summary = self._summarize_for_validation(inputs["fused_document"])
        
        # Get confidence scores
        ocr_conf_data = inputs["text_analysis"].get("ocr_confidence_summary", {})
        ocr_confidence = ocr_conf_data.get("average_confidence", 0.7) if isinstance(ocr_conf_data, dict) else 0.7
        layout_confidence = inputs["vision_analysis"].get("spatial_confidence", 0.7)
        fusion_confidence = inputs["fused_document"].get("fusion_confidence", 0.7)
        
        # Prepare summaries
        ocr_summary = f"Average OCR confidence: {ocr_confidence:.2f}"
        if isinstance(ocr_conf_data, dict) and "confidence_samples" in ocr_conf_data:
            ocr_summary += f" (based on {ocr_conf_data['confidence_samples']} samples)"
        
        layout_summary = f"Layout analysis confidence: {layout_confidence:.2f}"
        fusion_summary_text = f"Fusion confidence: {fusion_confidence:.2f}"
        
        # Call LLM
        prompt = VALIDATION_AGENT_PROMPT_TEMPLATE.format(
            fused_document=fused_summary,
            ocr_confidence_summary=ocr_summary,
            layout_confidence_summary=layout_summary,
            fusion_confidence=fusion_summary_text
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
            
            validation = self._parse_validation_response(response, inputs)
            
            logger.info(f"🎉 Validation Agent completed for {document_id}")
            logger.info(f"   Overall confidence: {validation.get('overall_confidence', 0.0):.2f}")
            
            passed = validation.get('validation_passed', [])
            failed = validation.get('validation_failed', [])
            logger.info(f"   Passed checks: {len(passed)}")
            logger.info(f"   Failed checks: {len(failed)}")
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Validation Agent failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_basic_validation(inputs)
    
    def _summarize_for_validation(self, fused_document: Dict[str, Any]) -> str:
        """Create summary for validation."""
        try:
            summary_parts = []
            
            doc_type = fused_document.get("unified_structure", {}).get("document_type", 
                       fused_document.get("document_type", "unknown"))
            summary_parts.append(f"Document Type: {doc_type}")
            
            fusion_conf = fused_document.get("fusion_confidence", 0.0)
            summary_parts.append(f"Fusion Confidence: {fusion_conf:.2f}")
            
            extractions = fused_document.get("fused_extractions", {})
            if extractions:
                summary_parts.append(f"Extracted {len(extractions)} fields:")
                for key, value_info in list(extractions.items())[:10]:
                    if isinstance(value_info, dict):
                        value = value_info.get("value", str(value_info))
                        conf = value_info.get("multi_modal_confidence", 0.0)
                        summary_parts.append(f"  - {key}: {value} (conf: {conf:.2f})")
                    else:
                        summary_parts.append(f"  - {key}: {value_info}")
            
            structure = fused_document.get("unified_structure", {})
            if structure and isinstance(structure, dict):
                summary_parts.append("Document Structure:")
                for key, value in structure.items():
                    if key != "pages" and key != "document_type":
                        summary_parts.append(f"  - {key}: {value}")
            
            return "\n".join(summary_parts)
        except Exception as e:
            logger.warning(f"Could not create summary: {e}")
            return str(fused_document)[:2000]
    
    def _parse_validation_response(self, response: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with PRODUCTION JSON handling."""
        try:
            cleaned_response = response.strip()
            
            # Remove markdown
            if "```json" in cleaned_response:
                cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
                cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
            elif "```" in cleaned_response:
                cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # CRITICAL FIX
                json_str = self._fix_json(json_str)
                
                try:
                    validation = json.loads(json_str)
                    logger.info(f"✅ Successfully parsed JSON from Validation LLM")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                    return self._create_basic_validation(inputs)
            else:
                logger.warning("⚠️ No JSON found in Validation LLM response")
                return self._create_basic_validation(inputs)
            
            # Calculate overall confidence if missing
            if "overall_confidence" not in validation:
                vision_conf = inputs["vision_analysis"].get("spatial_confidence", 0.0)
                text_conf = inputs["text_analysis"].get("semantic_confidence", 0.0)
                fusion_conf = inputs["fused_document"].get("fusion_confidence", 0.0)
                overall_conf = (vision_conf * 0.2 + text_conf * 0.3 + fusion_conf * 0.5)
                validation["overall_confidence"] = round(overall_conf, 3)
            
            # Add metadata
            validation["agent"] = self.name
            validation["document_id"] = inputs["document_id"]
            validation["timestamp"] = datetime.now().isoformat()
            
            # Ensure required fields
            for key in ["validation_passed", "validation_failed", "explainability_notes"]:
                if key not in validation or not isinstance(validation[key], list):
                    validation[key] = []
            
            if "hallucination_risk" not in validation:
                overall_conf = validation.get("overall_confidence", 0.5)
                extraction_count = len(inputs["fused_document"].get("fused_extractions", {}))
                
                if overall_conf > 0.8 and extraction_count > 3:
                    hallucination_risk = 0.1
                elif overall_conf > 0.6:
                    hallucination_risk = 0.3
                else:
                    hallucination_risk = 0.5
                
                validation["hallucination_risk"] = hallucination_risk
            
            # Limit list lengths
            for key in ["validation_passed", "validation_failed"]:
                if validation[key] and len(validation[key]) > 10:
                    validation[key] = validation[key][:10]
            
            # Clean structure
            validation = self._clean_structure(validation)
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Failed to parse Validation Agent response: {e}")
            return self._create_basic_validation(inputs)
    
    def _fix_json(self, json_str: str) -> str:
        """Fix all common JSON issues."""
        json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        json_str = re.sub(r'\s+', ' ', json_str).strip()
        
        # Critical fix
        json_str = re.sub(r'"+([^"]+)"+', lambda m: f'"{m.group(1)}"' if m.group(0).count('"') > 2 else m.group(0), json_str)
        
        json_str = json_str.replace('\\"', '"')
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        json_str = re.sub(r'([,{]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', json_str)
        
        def fix_value(m):
            colon, val, after = m.groups()
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or \
               re.match(r'^-?\d+(\.\d+)?$', val) or \
               val in ['true', 'false', 'null'] or \
               val.startswith('[') or val.startswith('{'):
                return f'{colon}{val}{after}'
            return f'{colon}"{val}"{after}'
        
        json_str = re.sub(r'(:\s*)([^,}\]]+?)(\s*[,}\]])', fix_value, json_str)
        
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'
        
        json_str = re.sub(r',\s*,', ',', json_str)
        
        return json_str
    
    def _clean_structure(self, data: Any) -> Any:
        """Recursively clean data."""
        if isinstance(data, dict):
            return {k: self._clean_structure(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_structure(item) for item in data]
        elif isinstance(data, str):
            s = data.strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ['"', "'"]:
                s = s[1:-1]
            return re.sub(r'\s+', ' ', s)
        return data
    
    def _create_basic_validation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic validation."""
        fused_doc = inputs.get("fused_document", {})
        vision_analysis = inputs.get("vision_analysis", {})
        text_analysis = inputs.get("text_analysis", {})
        
        vision_conf = vision_analysis.get("spatial_confidence", 0.0)
        text_conf = text_analysis.get("semantic_confidence", 0.0)
        fusion_conf = fused_doc.get("fusion_confidence", 0.0)
        
        overall_conf = (vision_conf * 0.3 + text_conf * 0.3 + fusion_conf * 0.4)
        
        validation_passed = []
        validation_failed = []
        
        if fusion_conf > 0.5:
            validation_passed.append("Fusion confidence above threshold (0.5)")
        else:
            validation_failed.append(f"Low fusion confidence: {fusion_conf:.2f}")
        
        if text_conf > 0.6:
            validation_passed.append("Text analysis confidence acceptable (>0.6)")
        else:
            validation_failed.append(f"Low text analysis confidence: {text_conf:.2f}")
        
        extractions = fused_doc.get("fused_extractions", {})
        if extractions and isinstance(extractions, dict) and len(extractions) > 0:
            validation_passed.append(f"Extracted {len(extractions)} fields")
        else:
            validation_failed.append("No fields extracted")
        
        doc_type = fused_doc.get("unified_structure", {}).get("document_type", 
                  text_analysis.get("document_type", "unknown"))
        if doc_type != "unknown":
            validation_passed.append(f"Identified as {doc_type}")
        else:
            validation_failed.append("Could not identify document type")
        
        alignment_notes = fused_doc.get("alignment_notes", [])
        if alignment_notes and isinstance(alignment_notes, list) and len(alignment_notes) > 0:
            validation_passed.append("Vision and text analyses aligned")
        
        if overall_conf > 0.7 and len(validation_passed) > len(validation_failed):
            hallucination_risk = 0.2
        elif overall_conf > 0.5:
            hallucination_risk = 0.4
        else:
            hallucination_risk = 0.6
        
        return {
            "agent": self.name,
            "document_id": inputs.get("document_id", ""),
            "overall_confidence": round(overall_conf, 3),
            "validation_passed": validation_passed,
            "validation_failed": validation_failed,
            "hallucination_risk": hallucination_risk,
            "explainability_notes": [
                "Basic validation using confidence scores",
                f"Vision confidence: {vision_conf:.2f}",
                f"Text confidence: {text_conf:.2f}",
                f"Fusion confidence: {fusion_conf:.2f}",
                f"Overall confidence: {overall_conf:.2f}",
                f"Document type: {doc_type}"
            ],
            "timestamp": datetime.now().isoformat()
        }
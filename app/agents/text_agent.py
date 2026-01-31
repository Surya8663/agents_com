"""
Text Intelligence Agent - COMPLETE PRODUCTION FIX
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
        """Extract semantic information from OCR."""
        if not self.validate_inputs(inputs, ["document_id", "ocr_data"]):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        ocr_data = inputs["ocr_data"]
        
        logger.info(f"🔤 Text Agent processing document {document_id}")
        
        ocr_summary = self._summarize_ocr(ocr_data)
        confidence_summary = self._calculate_confidence_summary(ocr_data)
        
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
                return self._create_fallback_analysis(ocr_data, document_id, confidence_summary, "LLM empty response")
            
            logger.info(f"✅ LLM response received: {len(response)} chars")
            
            analysis = self._parse_text_response(response, ocr_data, confidence_summary)
            
            logger.info(f"🎉 Text Agent completed for {document_id}")
            logger.info(f"   Document type: {analysis.get('document_type', 'unknown')}")
            
            kv_pairs = analysis.get("key_value_pairs", {})
            if kv_pairs:
                logger.info(f"   Extracted {len(kv_pairs)} key-value pairs")
                for key, value in list(kv_pairs.items())[:3]:
                    logger.info(f"     • {key}: {value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Text Agent failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_analysis(ocr_data, document_id, confidence_summary, str(e))
    
    def _summarize_ocr(self, ocr_data: Dict[str, Any]) -> str:
        """Create text summary of OCR for LLM."""
        summary_parts = []
        
        if "pages" in ocr_data:
            for page in ocr_data["pages"]:
                page_num = page.get("page_number", 1)
                regions = page.get("regions", [])
                
                page_summary = f"Page {page_num}: {len(regions)} OCR regions"
                text_count = 0
                for i, region in enumerate(regions):
                    text = region.get("ocr_text", "").strip()
                    confidence = region.get("ocr_confidence", 0.0)
                    
                    if text:
                        text_count += 1
                        if text_count <= 5:
                            preview = text[:100] + "..." if len(text) > 100 else text
                            page_summary += f"\n  Region {i} ({confidence:.2f}): {preview}"
                
                if text_count > 5:
                    page_summary += f"\n  ... and {text_count - 5} more text regions"
                
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
                    text = region.get("ocr_text", "").strip()
                    
                    if confidence > 0:
                        confidences.append(confidence)
                    if text:
                        text_regions += 1
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
        else:
            avg_confidence = 0.7
            max_confidence = 0.9
            min_confidence = 0.5
        
        return {
            "average_confidence": round(avg_confidence, 3),
            "max_confidence": round(max_confidence, 3),
            "min_confidence": round(min_confidence, 3),
            "total_regions": total_regions,
            "text_regions": text_regions,
            "confidence_samples": len(confidences)
        }
    
    def _parse_text_response(self, response: str, ocr_data: Dict[str, Any], 
                           confidence_summary: Dict[str, Any]) -> Dict[str, Any]:
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
            if not json_match:
                logger.warning("⚠️ No JSON found in LLM response")
                return self._extract_from_text_fallback(cleaned_response, ocr_data, confidence_summary)
            
            json_str = json_match.group()
            
            # CRITICAL FIX
            json_str = self._fix_json(json_str)
            
            try:
                analysis = json.loads(json_str)
                logger.info(f"✅ Successfully parsed JSON from LLM response")
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error at position {e.pos}: {e.msg}")
                logger.error(f"Problematic JSON snippet: ...{json_str[max(0, e.pos-50):e.pos+50]}...")
                analysis = self._extract_partial_json(json_str)
            
            # Ensure required fields
            analysis = self._ensure_required_fields(analysis, ocr_data, confidence_summary)
            
            # Clean structure
            analysis = self._clean_structure(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to parse Text Agent response: {e}")
            return self._create_fallback_analysis(ocr_data, document_id=None, 
                                                confidence_summary=confidence_summary, 
                                                error=str(e))
    
    def _fix_json(self, json_str: str) -> str:
        """Fix all common JSON issues from LLMs."""
        # Normalize whitespace
        json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        json_str = re.sub(r'\s+', ' ', json_str).strip()
        
        # THE CRITICAL FIX: Remove extra quotes
        json_str = re.sub(r'"+([^"]+)"+', lambda m: f'"{m.group(1)}"' if m.group(0).count('"') > 2 else m.group(0), json_str)
        
        # Remove escaped quotes
        json_str = json_str.replace('\\"', '"')
        
        # Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Quote property names
        json_str = re.sub(r'([,{]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', json_str)
        
        # Quote unquoted values
        def fix_value(m):
            colon, val, after = m.groups()
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or \
               re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', val) or \
               val in ['true', 'false', 'null'] or \
               val.startswith('[') or val.startswith('{'):
                return f'{colon}{val}{after}'
            return f'{colon}"{val}"{after}'
        
        json_str = re.sub(r'(:\s*)([^,}\]]+?)(\s*[,}\]])', fix_value, json_str)
        
        # Fix arrays
        def fix_array(m):
            content = m.group(1).strip()
            if not content:
                return '[]'
            
            parts = []
            current = ''
            depth = 0
            in_str = False
            
            for ch in content:
                if ch == '"' and (not current or current[-1] != '\\'):
                    in_str = not in_str
                if not in_str:
                    if ch in '[{':
                        depth += 1
                    elif ch in ']}':
                        depth -= 1
                
                if ch == ',' and depth == 0 and not in_str:
                    parts.append(current.strip())
                    current = ''
                else:
                    current += ch
            
            if current.strip():
                parts.append(current.strip())
            
            fixed = []
            for p in parts:
                if not p:
                    continue
                if (p.startswith('"') and p.endswith('"')) or \
                   re.match(r'^-?\d+(\.\d+)?$', p) or \
                   p in ['true', 'false', 'null'] or \
                   p.startswith('[') or p.startswith('{'):
                    fixed.append(p)
                else:
                    fixed.append(f'"{p}"')
            
            return '[' + ', '.join(fixed) + ']'
        
        json_str = re.sub(r'\[([^\[\]]*?)\]', fix_array, json_str)
        
        # Ensure structure
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
    
    def _extract_partial_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial data from malformed JSON."""
        analysis = {}
        try:
            kv_pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
            matches = re.findall(kv_pattern, json_str)
            if matches:
                analysis["key_value_pairs"] = {k: v for k, v in matches}
                logger.info(f"📊 Extracted {len(analysis['key_value_pairs'])} key-value pairs from partial JSON")
            
            doc_type_pattern = r'"document_type"\s*:\s*"([^"]*)"'
            doc_type_match = re.search(doc_type_pattern, json_str)
            if doc_type_match:
                analysis["document_type"] = doc_type_match.group(1)
        except Exception as e:
            logger.warning(f"Could not extract partial JSON: {e}")
        return analysis
    
    def _extract_from_text_fallback(self, text: str, ocr_data: Dict[str, Any], 
                                  confidence_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Extract from plain text when JSON fails."""
        analysis = {
            "agent": self.name,
            "document_type": self._infer_document_type(ocr_data),
            "semantic_confidence": confidence_summary["average_confidence"],
            "ocr_confidence_summary": confidence_summary,
            "key_entities": {},
            "key_value_pairs": {},
            "table_extractions": [],
            "parsing_method": "text_fallback"
        }
        
        kv_pairs = {}
        patterns = [
            r'([A-Za-z\s]+)[:=]\s*([^\n]+)',
            r'"([^"]+)"\s*[:=]\s*"([^"]+)"',
            r'([A-Za-z\s]+)\s*=\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                key = key.strip().lower()
                value = value.strip()
                if key and value and len(key) < 50 and len(value) < 200:
                    kv_pairs[key] = value
        
        if kv_pairs:
            analysis["key_value_pairs"] = kv_pairs
            logger.info(f"📊 Extracted {len(kv_pairs)} key-value pairs from text fallback")
        
        return analysis
    
    def _ensure_required_fields(self, analysis: Dict[str, Any], ocr_data: Dict[str, Any],
                              confidence_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist."""
        if not isinstance(analysis, dict):
            analysis = {}
        
        analysis["agent"] = self.name
        
        if "document_type" not in analysis:
            analysis["document_type"] = self._infer_document_type(ocr_data)
        
        if "semantic_confidence" not in analysis:
            analysis["semantic_confidence"] = confidence_summary["average_confidence"]
        
        if "ocr_confidence_summary" not in analysis:
            analysis["ocr_confidence_summary"] = confidence_summary
        
        if "key_entities" not in analysis or not isinstance(analysis["key_entities"], dict):
            analysis["key_entities"] = {}
        
        if "key_value_pairs" not in analysis or not isinstance(analysis["key_value_pairs"], dict):
            analysis["key_value_pairs"] = {}
        
        if "table_extractions" not in analysis or not isinstance(analysis["table_extractions"], list):
            analysis["table_extractions"] = []
        
        return analysis
    
    def _create_fallback_analysis(self, ocr_data: Dict[str, Any], document_id: str,
                                confidence_summary: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create fallback analysis."""
        logger.info("Creating fallback text analysis")
        all_text = self._get_all_text(ocr_data)
        
        return {
            "agent": self.name,
            "document_id": document_id,
            "document_type": self._infer_document_type(ocr_data),
            "semantic_confidence": confidence_summary["average_confidence"],
            "ocr_confidence_summary": confidence_summary,
            "key_entities": self._extract_basic_entities(all_text),
            "key_value_pairs": self._extract_basic_key_values(all_text),
            "table_extractions": [],
            "fallback_mode": True,
            "error": error_msg,
            "text_preview": all_text[:500] + "..." if len(all_text) > 500 else all_text
        }
    
    def _get_all_text(self, ocr_data: Dict[str, Any]) -> str:
        """Extract all text from OCR."""
        all_text = []
        if "pages" in ocr_data:
            for page in ocr_data["pages"]:
                for region in page.get("regions", []):
                    text = region.get("ocr_text", "").strip()
                    if text:
                        all_text.append(text)
        return " ".join(all_text)
    
    def _infer_document_type(self, ocr_data: Dict[str, Any]) -> str:
        """Infer document type."""
        all_text = self._get_all_text(ocr_data).lower()
        
        doc_keywords = {
            "invoice": ["invoice", "bill", "payment", "amount due", "total"],
            "receipt": ["receipt", "purchase", "transaction", "paid"],
            "contract": ["agreement", "contract", "terms", "clause"],
            "report": ["report", "analysis", "findings", "summary"],
            "letter": ["dear", "sincerely", "yours"],
            "form": ["form", "application", "section"]
        }
        
        for doc_type, keywords in doc_keywords.items():
            if any(kw in all_text for kw in keywords):
                return doc_type
        
        return "unknown"
    
    def _extract_basic_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract basic entities."""
        return {
            "dates": list(set(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)))[:5],
            "amounts": list(set(re.findall(r'\$\s*\d+\.?\d*', text)))[:5],
            "emails": list(set(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)))[:5],
            "phones": list(set(re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)))[:5],
            "names": list(set(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)))[:5]
        }
    
    def _extract_basic_key_values(self, text: str) -> Dict[str, str]:
        """Extract basic key-value pairs."""
        kv_pairs = {}
        patterns = [
            (r'(invoice\s*(?:#|number)?)\s*[:=]?\s*([A-Za-z0-9\-]+)', 'invoice_number'),
            (r'(date|Date)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'date'),
            (r'(total|Total)\s*[:=]?\s*(\$\s*\d+\.?\d*)', 'total_amount'),
            (r'(name|Name)\s*[:=]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', 'name'),
            (r'(email|Email)\s*[:=]?\s*([\w\.-]+@[\w\.-]+\.\w+)', 'email')
        ]
        
        for pattern, key in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    kv_pairs[key] = match[1].strip()
        
        return kv_pairs
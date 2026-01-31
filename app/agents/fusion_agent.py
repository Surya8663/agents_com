"""
Fusion Intelligence Agent - ULTIMATE PRODUCTION FIX
Generates VALID JSON that won't break RAG indexing.
"""
import json
import logging
import re
import ast
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.llm.prompts import FUSION_AGENT_SYSTEM_PROMPT, FUSION_AGENT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class FusionAgent(BaseAgent):
    """Fuses vision and text analysis with RAG context."""
    
    def __init__(self):
        """Initialize Fusion Agent."""
        super().__init__("FusionAgent")
        try:
            from app.rag.retriever import retriever
            self.retriever = retriever
            self.has_rag = True
        except ImportError:
            logger.warning("RAG retriever not available")
            self.retriever = None
            self.has_rag = False
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse vision and text analysis with optional RAG context."""
        required = ["document_id", "vision_analysis", "text_analysis"]
        if not self.validate_inputs(inputs, required):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        logger.info(f"🤝 Fusion Agent processing document {document_id}")
        
        vision_summary = self._summarize_for_fusion(inputs["vision_analysis"])
        text_summary = self._summarize_for_fusion(inputs["text_analysis"])
        
        rag_context = ""
        if self.has_rag:
            rag_context = await self._retrieve_rag_context(document_id)
        else:
            rag_context = "RAG context not available."
        
        # ADD STRICT JSON INSTRUCTIONS TO PROMPT
        strict_json_prompt = self._build_fusion_prompt_with_json_instructions(
            vision_summary, text_summary, rag_context
        )
        
        logger.info(f"📡 Fusion Agent calling LLM with strict JSON instructions")
        
        try:
            response = await self.llm.generate(
                prompt=strict_json_prompt,
                system_prompt=FUSION_AGENT_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            if not response or len(response.strip()) == 0:
                logger.error("LLM returned empty response")
                return self._create_basic_fusion(inputs)
            
            logger.info(f"✅ LLM response received: {len(response)} chars")
            fusion = self._parse_fusion_response_with_validation(response, inputs, rag_context)
            
            # Log results
            fusion_conf = fusion.get('fusion_confidence', 0.0)
            logger.info(f"✅ Fusion Agent completed for {document_id}")
            logger.info(f"   Fusion confidence: {fusion_conf:.2f}")
            if fusion.get("fused_extractions"):
                logger.info(f"   Fused extractions: {len(fusion['fused_extractions'])}")
            
            return fusion
            
        except Exception as e:
            logger.error(f"❌ Fusion Agent failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_basic_fusion(inputs)
    
    def _build_fusion_prompt_with_json_instructions(self, vision_summary: str, text_summary: str, 
                                                  rag_context: str) -> str:
        """Build fusion prompt with STRICT JSON instructions."""
        base_prompt = FUSION_AGENT_PROMPT_TEMPLATE.format(
            vision_analysis=vision_summary,
            text_analysis=text_summary
        )
        
        # ADD STRICT JSON FORMATTING INSTRUCTIONS
        json_instructions = """

CRITICAL JSON FORMATTING RULES - MUST FOLLOW:
1. ALL string values MUST be in double quotes: "value" NOT value
2. ALL property names MUST be in double quotes: "property_name"
3. ALL string references MUST be quoted: "parent": "p3_r0" NOT "parent": p3_r0
4. NO trailing commas: remove comma before ] or }
5. Escape quotes inside strings: "text with \"quotes\" inside"
6. Use null for empty values, not None or undefined
7. NO comments in JSON
8. Validate your JSON is valid before returning

VALID JSON EXAMPLE:
{
  "fusion_confidence": 0.85,
  "unified_structure": {
    "document_type": "invoice",
    "pages": [1, 2, 3],
    "rag_informed": true
  },
  "fused_extractions": {
    "invoice_number": {
      "value": "INV-2023-001",
      "multi_modal_confidence": 0.9,
      "source": "text_agent",
      "has_rag_context": false
    }
  },
  "conflict_resolutions": [],
  "alignment_notes": ["Vision and text analyses aligned"],
  "hierarchy_analysis": {
    "parent_children_map": {
      "p3_r0": {"parent": null},
      "p2_r0": {"parent": "p3_r0"}  # NOTE: "p3_r0" is quoted!
    }
  }
}

INVALID JSON EXAMPLE (WILL BREAK SYSTEM):
{
  "fusion_confidence": 0.85,
  "unified_structure": {
    "document_type": invoice,  # ERROR: no quotes
    "pages": [1, 2, 3],
    "rag_informed": true
  },
  "hierarchy_analysis": {
    "parent_children_map": {
      "p3_r0": {"parent": null},
      "p2_r0": {"parent": p3_r0}  # ERROR: p3_r0 not quoted!
    }
  }
}

YOUR RESPONSE MUST BE VALID JSON. Start with { and end with }
"""
        
        prompt = base_prompt + json_instructions
        
        if rag_context and "not available" not in rag_context.lower():
            prompt += f"\n\nADDITIONAL CONTEXT FROM SIMILAR DOCUMENTS:\n{rag_context}\n\n"
            prompt += "Use this additional context to improve your analysis.\n"
        
        return prompt
    
    def _parse_fusion_response_with_validation(self, response: str, inputs: Dict[str, Any], 
                                             rag_context: str) -> Dict[str, Any]:
        """Parse LLM response with STRICT validation and automatic fixing."""
        try:
            # Step 1: Extract JSON from response
            json_str = self._extract_and_clean_json(response)
            
            if not json_str:
                logger.error("❌ No JSON found in response")
                return self._create_basic_fusion(inputs)
            
            # Step 2: Parse with validation
            fusion = self._parse_json_with_validation(json_str)
            
            if not fusion:
                logger.error("❌ Failed to parse JSON after validation")
                return self._create_basic_fusion(inputs)
            
            # Step 3: Validate JSON structure
            if not self._validate_json_structure(fusion):
                logger.warning("⚠️ JSON structure validation failed, fixing...")
                fusion = self._fix_json_structure(fusion)
            
            # Step 4: Post-process
            fusion = self._post_process_fusion(fusion, inputs, rag_context)
            
            # Step 5: Final validation
            self._final_json_validation(fusion)
            
            return fusion
            
        except Exception as e:
            logger.error(f"❌ Failed to parse Fusion Agent response: {e}")
            return self._create_basic_fusion(inputs)
    
    def _extract_and_clean_json(self, response: str) -> str:
        """Extract and clean JSON from LLM response."""
        # Remove markdown code blocks
        response = response.strip()
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        response = re.sub(r'```', '', response)
        
        # Find JSON object
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON object found in response")
            # Try to find the start of JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
            else:
                return ""
        else:
            json_str = json_match.group(1)
        
        # Apply CRITICAL fixes
        json_str = self._apply_critical_json_fixes(json_str)
        
        return json_str
    
    def _apply_critical_json_fixes(self, json_str: str) -> str:
        """Apply CRITICAL JSON fixes that are breaking RAG."""
        if not json_str:
            return "{}"
        
        # FIX 1: Quote unquoted string values like "parent": p3_r0
        # This is the MAIN FIX for your error
        def fix_unquoted_refs(match):
            key = match.group(1)
            value = match.group(2)
            
            # Check if value is a reference like p3_r0, p2_r0, etc.
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', value) and value not in ['true', 'false', 'null']:
                return f'"{key}": "{value}"'
            return f'"{key}": {value}'
        
        # Pattern for key: value where value might be unquoted reference
        json_str = re.sub(r'"([^"]+)"\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)(?=[,\]}])', 
                         fix_unquoted_refs, json_str)
        
        # FIX 2: Fix all string values (quotes missing around string values)
        def fix_string_values(match):
            key = match.group(1)
            value = match.group(2)
            
            # Already quoted or is number/boolean/null
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")) or \
               re.match(r'^-?\d+(\.\d+)?$', value) or \
               value in ['true', 'false', 'null']:
                return f'"{key}": {value}'
            
            # It's a string - quote it
            # Escape any quotes inside
            value = value.replace('"', '\\"')
            return f'"{key}": "{value}"'
        
        # More comprehensive pattern
        json_str = re.sub(r'"([^"]+)"\s*:\s*([^,\[\]{}\s"\'][^,\[\]{}"\']*)(?=[,\]}])', 
                         fix_string_values, json_str)
        
        # FIX 3: Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # FIX 4: Escape quotes inside strings
        def escape_inner_quotes(match):
            content = match.group(1)
            # Escape unescaped quotes
            content = re.sub(r'(?<!\\)"', r'\"', content)
            return f'"{content}"'
        
        json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_inner_quotes, json_str)
        
        # FIX 5: Remove control characters that break JSON
        json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
        
        # FIX 6: Ensure proper JSON structure
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'
        
        return json_str
    
    def _parse_json_with_validation(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON with multiple validation attempts."""
        # Attempt 1: Standard JSON parse
        try:
            parsed = json.loads(json_str)
            logger.info("✅ JSON parsed successfully on first attempt")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ First parse failed: {e.msg} at position {e.pos}")
            
            # Show context of error
            start = max(0, e.pos - 50)
            end = min(len(json_str), e.pos + 50)
            logger.debug(f"Error context: ...{json_str[start:end]}...")
        
        # Attempt 2: Try with more aggressive fixing
        try:
            # Apply advanced fixes
            fixed_json = self._advanced_json_repair(json_str)
            parsed = json.loads(fixed_json)
            logger.info("✅ JSON parsed after advanced repair")
            return parsed
        except Exception as e:
            logger.warning(f"⚠️ Advanced repair failed: {e}")
        
        # Attempt 3: Use ast.literal_eval as last resort
        try:
            import ast
            # Convert JSON-like to Python dict
            python_str = json_str.replace(': ', ':').replace(', ', ',')
            python_str = python_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            
            # Parse as Python literal
            parsed = ast.literal_eval(python_str)
            
            # Convert back to proper types
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                elif obj is None:
                    return None
                elif isinstance(obj, bool):
                    return obj
                elif isinstance(obj, (int, float)):
                    return obj
                else:
                    return str(obj)
            
            parsed = convert_types(parsed)
            logger.info("✅ JSON parsed using ast.literal_eval")
            return parsed
        except Exception as e:
            logger.error(f"❌ All parsing attempts failed: {e}")
            return {}
    
    def _advanced_json_repair(self, json_str: str) -> str:
        """Advanced JSON repair for stubborn cases."""
        # Fix nested unquoted values
        depth = 0
        in_string = False
        result = []
        i = 0
        
        while i < len(json_str):
            char = json_str[i]
            
            if char == '"' and (i == 0 or json_str[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif not in_string:
                if char in '[{':
                    depth += 1
                    result.append(char)
                elif char in '}]':
                    depth -= 1
                    result.append(char)
                elif char == ':' and i + 1 < len(json_str):
                    result.append(char)
                    # Look ahead for value
                    j = i + 1
                    while j < len(json_str) and json_str[j] in ' \t\n\r':
                        j += 1
                    
                    if j < len(json_str):
                        next_char = json_str[j]
                        if next_char not in ['"', '[', '{', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            # Likely an unquoted string
                            result.append('"')
                            # Find end of value
                            k = j
                            while k < len(json_str) and json_str[k] not in [',', '}', ']', '\n', '\r']:
                                result.append(json_str[k])
                                k += 1
                            result.append('"')
                            i = k - 1
                        else:
                            result.append(next_char)
                            i = j
                else:
                    result.append(char)
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def _validate_json_structure(self, fusion: Dict[str, Any]) -> bool:
        """Validate JSON structure."""
        try:
            # Try to serialize and deserialize
            json_str = json.dumps(fusion, ensure_ascii=False)
            parsed = json.loads(json_str)
            return isinstance(parsed, dict)
        except Exception as e:
            logger.warning(f"JSON structure validation failed: {e}")
            return False
    
    def _fix_json_structure(self, fusion: Dict[str, Any]) -> Dict[str, Any]:
        """Fix JSON structure issues."""
        def fix_value(value):
            if isinstance(value, dict):
                return {k: fix_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [fix_value(item) for item in value]
            elif isinstance(value, str):
                # Ensure string doesn't break JSON
                value = value.replace('\n', ' ').replace('\r', ' ')
                value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
                return value
            elif isinstance(value, (int, float, bool)):
                return value
            elif value is None:
                return None
            else:
                # Convert to string
                return str(value)
        
        return fix_value(fusion)
    
    def _final_json_validation(self, fusion: Dict[str, Any]):
        """Final JSON validation before returning."""
        try:
            json_str = json.dumps(fusion, ensure_ascii=False)
            # Test that it can be parsed back
            json.loads(json_str)
            logger.info("✅ Final JSON validation passed")
        except Exception as e:
            logger.error(f"❌ Final JSON validation failed: {e}")
            raise
    
    def _summarize_for_fusion(self, analysis: Dict[str, Any]) -> str:
        """Summarize analysis for fusion prompt."""
        if not analysis or not isinstance(analysis, dict):
            return "No analysis data available"
        
        summary_parts = []
        agent_name = analysis.get("agent", "Unknown")
        summary_parts.append(f"Analysis from {agent_name}:")
        
        if "spatial_confidence" in analysis:
            summary_parts.append(f"  - Spatial confidence: {analysis['spatial_confidence']}")
        if "semantic_confidence" in analysis:
            summary_parts.append(f"  - Semantic confidence: {analysis['semantic_confidence']}")
        if "document_type" in analysis:
            summary_parts.append(f"  - Document type: {analysis['document_type']}")
        
        if "key_value_pairs" in analysis and analysis["key_value_pairs"]:
            summary_parts.append(f"  - Key-value pairs found: {len(analysis['key_value_pairs'])}")
            for key, value in list(analysis["key_value_pairs"].items())[:5]:
                if isinstance(value, dict):
                    value = value.get("value", value)
                summary_parts.append(f"    • {key}: {value}")
        
        if "key_entities" in analysis and analysis["key_entities"]:
            summary_parts.append(f"  - Entities found:")
            for entity_type, entities in list(analysis["key_entities"].items())[:3]:
                if entities:
                    summary_parts.append(f"    • {entity_type}: {entities[:3]}")
        
        return "\n".join(summary_parts)
    
    async def _retrieve_rag_context(self, document_id: str) -> str:
        """Retrieve RAG context."""
        if not self.has_rag or not self.retriever:
            return "RAG context not available."
        
        try:
            from app.rag.multimodal_schema import RAGQuery
            rag_query = RAGQuery(
                query="What type of document is this?",
                document_id=document_id,
                top_k=2,
                similarity_threshold=0.6
            )
            response = await self.retriever.query(rag_query)
            if response.answer and response.confidence > 0.5:
                return f"RAG Context: This appears to be a {response.answer}"
            else:
                return "No RAG context available for this document type."
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
            return "RAG context unavailable."
    
    def _post_process_fusion(self, fusion: Dict[str, Any], inputs: Dict[str, Any], rag_context: str) -> Dict[str, Any]:
        """Post-process and clean fusion data."""
        fusion["agent"] = self.name
        fusion["document_id"] = inputs["document_id"]
        fusion["rag_context_used"] = rag_context and "not available" not in rag_context.lower()
        
        # Ensure fusion_confidence
        if "fusion_confidence" not in fusion:
            vision_conf = float(inputs["vision_analysis"].get("spatial_confidence", 0.5))
            text_conf = float(inputs["text_analysis"].get("semantic_confidence", 0.5))
            rag_boost = 0.1 if fusion.get("rag_context_used") else 0.0
            fusion["fusion_confidence"] = min(1.0, (vision_conf * 0.4 + text_conf * 0.6) + rag_boost)
        else:
            try:
                fusion["fusion_confidence"] = float(fusion["fusion_confidence"])
            except (ValueError, TypeError):
                fusion["fusion_confidence"] = 0.5
        
        # Ensure required fields
        if "unified_structure" not in fusion:
            fusion["unified_structure"] = {
                "document_type": inputs["text_analysis"].get("document_type", "unknown"),
                "pages": [],
                "rag_informed": fusion.get("rag_context_used", False)
            }
        
        if "fused_extractions" not in fusion:
            fusion["fused_extractions"] = {}
        
        if "conflict_resolutions" not in fusion:
            fusion["conflict_resolutions"] = []
        
        if "alignment_notes" not in fusion:
            fusion["alignment_notes"] = ["Vision and text analysis aligned"]
        
        # Traceability
        fusion["traceability"] = {
            "vision_confidence": float(inputs["vision_analysis"].get("spatial_confidence", 0.0)),
            "text_confidence": float(inputs["text_analysis"].get("semantic_confidence", 0.0)),
            "ocr_confidence": float(inputs["text_analysis"].get("ocr_confidence_summary", {}).get("average_confidence", 0.0)),
            "rag_context_used": fusion.get("rag_context_used", False),
            "agent_versions": {
                "vision": inputs["vision_analysis"].get("agent", "unknown"),
                "text": inputs["text_analysis"].get("agent", "unknown"),
                "fusion": self.name
            }
        }
        
        # Clean all strings recursively
        fusion = self._clean_structure(fusion)
        
        return fusion
    
    def _clean_structure(self, data: Any) -> Any:
        """Recursively clean data structure."""
        if isinstance(data, dict):
            return {k: self._clean_structure(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_structure(item) for item in data]
        elif isinstance(data, str):
            # Remove extra quotes and normalize
            s = data.strip()
            # Remove surrounding quotes
            while len(s) >= 2 and s[0] == s[-1] and s[0] in ['"', "'"]:
                s = s[1:-1]
            # Normalize whitespace
            s = re.sub(r'\s+', ' ', s)
            # Escape problematic characters
            s = s.replace('\n', ' ').replace('\r', ' ')
            return s
        elif isinstance(data, (int, float, bool)):
            return data
        elif data is None:
            return None
        else:
            # Convert any other type to string
            return str(data)
    
    def _create_basic_fusion(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic fusion when LLM fails."""
        logger.info("Creating basic fusion analysis (LLM unavailable)")
        
        vision_analysis = inputs.get("vision_analysis", {})
        text_analysis = inputs.get("text_analysis", {})
        
        vision_conf = float(vision_analysis.get("spatial_confidence", 0.5))
        text_conf = float(text_analysis.get("semantic_confidence", 0.5))
        fusion_conf = (vision_conf * 0.4 + text_conf * 0.6)
        
        text_extractions = text_analysis.get("key_value_pairs", {})
        fused_extractions = {}
        
        for key, value in text_extractions.items():
            if isinstance(value, dict):
                actual_value = value.get("value", value)
            else:
                actual_value = value
            
            # Clean the value to ensure JSON compatibility
            cleaned_value = str(actual_value).replace('"', "'").replace('\n', ' ').replace('\r', ' ')
            
            fused_extractions[key] = {
                "value": cleaned_value,
                "multi_modal_confidence": text_conf,
                "source": "text_agent",
                "has_rag_context": False,
                "extraction_method": "fallback"
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
            "alignment_notes": ["Basic fusion - using text analysis data"],
            "rag_context_used": False,
            "traceability": {
                "vision_confidence": vision_conf,
                "text_confidence": text_conf,
                "ocr_confidence": float(text_analysis.get("ocr_confidence_summary", {}).get("average_confidence", 0.0)),
                "rag_context_used": False,
                "agent_versions": {
                    "vision": vision_analysis.get("agent", "unknown"),
                    "text": text_analysis.get("agent", "unknown"),
                    "fusion": self.name
                }
            },
            "fallback_mode": True
        }
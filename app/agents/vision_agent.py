"""
Vision Intelligence Agent - ULTIMATE PRODUCTION FIX
Generates VALID JSON that won't break RAG indexing.
"""
import json
import logging
import re
import ast
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
        """Analyze document layout."""
        if not self.validate_inputs(inputs, ["document_id", "layout_data"]):
            return {"error": "Invalid inputs"}
        
        document_id = inputs["document_id"]
        layout_data = inputs["layout_data"]
        
        logger.info(f"👁️ Vision Agent processing document {document_id}")
        
        pages_summary = self._summarize_layout(layout_data)
        
        # ADD STRICT JSON INSTRUCTIONS TO PROMPT
        strict_json_prompt = self._build_vision_prompt_with_json_instructions(pages_summary)
        
        logger.info(f"📡 Vision Agent calling LLM with strict JSON instructions")
        
        try:
            response = await self.llm.generate(
                prompt=strict_json_prompt,
                system_prompt=VISION_AGENT_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            if not response or len(response.strip()) == 0:
                logger.error("LLM returned empty response")
                return self._create_basic_analysis(layout_data, document_id)
            
            logger.info(f"✅ LLM response received: {len(response)} chars")
            
            analysis = self._parse_vision_response_with_validation(response, layout_data, document_id)
            
            logger.info(f"🎉 Vision Agent completed for {document_id}")
            logger.info(f"   Spatial confidence: {analysis.get('spatial_confidence', 0.0):.2f}")
            logger.info(f"   Reading order regions: {len(analysis.get('logical_reading_order', []))}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Vision Agent failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_basic_analysis(layout_data, document_id)
    
    def _build_vision_prompt_with_json_instructions(self, pages_summary: str) -> str:
        """Build vision prompt with STRICT JSON instructions."""
        base_prompt = VISION_AGENT_PROMPT_TEMPLATE.format(pages_summary=pages_summary)
        
        # ADD STRICT JSON FORMATTING INSTRUCTIONS
        json_instructions = """

CRITICAL JSON FORMATTING RULES - MUST FOLLOW:
1. ALL string values MUST be in double quotes: "value" NOT value
2. ALL property names MUST be in double quotes: "property_name"
3. ALL region references MUST be quoted: "parent": "p3_r0" NOT "parent": p3_r0
4. NO trailing commas: remove comma before ] or }
5. Escape quotes inside strings: "text with \"quotes\" inside"
6. Use null for empty values, not None or undefined
7. NO comments in JSON
8. Validate your JSON is valid before returning

VALID JSON EXAMPLE:
{
  "spatial_confidence": 0.85,
  "logical_reading_order": ["p1_r0", "p1_r1", "p1_r2"],
  "hierarchy_analysis": {
    "document_structure": "multi-page",
    "total_pages": 4,
    "regions_per_page": {
      "page_1": 5,
      "page_2": 3
    },
    "parent_children_map": {
      "p3_r0": {"parent": null},
      "p2_r0": {"parent": "p3_r0"}  # NOTE: "p3_r0" is quoted!
    }
  },
  "region_importance_scores": {
    "p1_r0": 0.9,
    "p1_r1": 0.7
  },
  "visual_validation_notes": ["Layout analysis completed successfully"]
}

INVALID JSON EXAMPLE (WILL BREAK SYSTEM):
{
  "spatial_confidence": 0.85,
  "logical_reading_order": ["p1_r0", "p1_r1", "p1_r2"],
  "hierarchy_analysis": {
    "document_structure": "multi-page",
    "total_pages": 4,
    "regions_per_page": {
      "page_1": 5,
      "page_2": 3
    },
    "parent_children_map": {
      "p3_r0": {"parent": null},
      "p2_r0": {"parent": p3_r0}  # ERROR: p3_r0 not quoted!
    }
  }
}

YOUR RESPONSE MUST BE VALID JSON. Start with { and end with }
"""
        
        return base_prompt + json_instructions
    
    def _parse_vision_response_with_validation(self, response: str, layout_data: Dict[str, Any], 
                                             document_id: str) -> Dict[str, Any]:
        """Parse LLM response with STRICT validation and automatic fixing."""
        try:
            # Step 1: Extract JSON from response
            json_str = self._extract_and_clean_json(response)
            
            if not json_str:
                logger.error("❌ No JSON found in response")
                return self._create_basic_analysis(layout_data, document_id)
            
            # Step 2: Parse with validation
            analysis = self._parse_json_with_validation(json_str)
            
            if not analysis:
                logger.error("❌ Failed to parse JSON after validation")
                return self._create_basic_analysis(layout_data, document_id)
            
            # Step 3: Validate JSON structure
            if not self._validate_json_structure(analysis):
                logger.warning("⚠️ JSON structure validation failed, fixing...")
                analysis = self._fix_json_structure(analysis)
            
            # Step 4: Post-process
            analysis = self._post_process_analysis(analysis, layout_data, document_id)
            
            # Step 5: Final validation
            self._final_json_validation(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to parse Vision Agent response: {e}")
            return self._create_basic_analysis(layout_data, document_id)
    
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
        
        # Apply CRITICAL fixes - SPECIFICALLY FOR VISION AGENT
        json_str = self._apply_critical_json_fixes_for_vision(json_str)
        
        return json_str
    
    def _apply_critical_json_fixes_for_vision(self, json_str: str) -> str:
        """Apply CRITICAL JSON fixes that are breaking RAG - SPECIFIC FOR VISION AGENT."""
        if not json_str:
            return "{}"
        
        # FIX 1: Quote unquoted region references like "parent": p3_r0
        # This is the MAIN FIX for your error
        def fix_unquoted_region_refs(match):
            key = match.group(1)
            value = match.group(2)
            
            # Check if value is a region reference like p3_r0, p2_r0, p1_r0, etc.
            if re.match(r'^p\d+_r\d+$', value):
                return f'"{key}": "{value}"'
            # Check if value is a general reference (alphanumeric with underscores)
            elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', value) and value not in ['true', 'false', 'null']:
                return f'"{key}": "{value}"'
            return f'"{key}": {value}'
        
        # Pattern for key: value where value might be unquoted region reference
        json_str = re.sub(r'"([^"]+)"\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)(?=[,\]}])', 
                         fix_unquoted_region_refs, json_str)
        
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
            python_str = json_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            
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
                            # Likely an unquoted string - quote it
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
    
    def _validate_json_structure(self, analysis: Dict[str, Any]) -> bool:
        """Validate JSON structure."""
        try:
            # Try to serialize and deserialize
            json_str = json.dumps(analysis, ensure_ascii=False)
            parsed = json.loads(json_str)
            return isinstance(parsed, dict)
        except Exception as e:
            logger.warning(f"JSON structure validation failed: {e}")
            return False
    
    def _fix_json_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
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
        
        return fix_value(analysis)
    
    def _final_json_validation(self, analysis: Dict[str, Any]):
        """Final JSON validation before returning."""
        try:
            json_str = json.dumps(analysis, ensure_ascii=False)
            # Test that it can be parsed back
            json.loads(json_str)
            logger.info("✅ Final JSON validation passed")
        except Exception as e:
            logger.error(f"❌ Final JSON validation failed: {e}")
            raise
    
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
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    elif isinstance(bbox, dict):
                        x1 = bbox.get('x1', 0)
                        y1 = bbox.get('y1', 0)
                        x2 = bbox.get('x2', 0)
                        y2 = bbox.get('y2', 0)
                    else:
                        x1, y1, x2, y2 = 0, 0, 0, 0
                    
                    confidence = det.get("confidence", 0.0)
                    label = det.get("label", det.get("type", "unknown"))
                    page_summary += f"\n  {region_id}: {label} at ({x1:.0f}, {y1:.0f}) - confidence: {confidence:.2f}"
                
                summary_parts.append(page_summary)
        
        return "\n\n".join(summary_parts) if summary_parts else "No layout data available"
    
    def _post_process_analysis(self, analysis: Dict[str, Any], layout_data: Dict[str, Any], 
                              document_id: str) -> Dict[str, Any]:
        """Post-process and clean analysis data."""
        analysis["agent"] = self.name
        analysis["document_id"] = document_id
        
        # Ensure required fields
        if "logical_reading_order" not in analysis:
            analysis["logical_reading_order"] = self._infer_reading_order(layout_data)
        
        if "spatial_confidence" not in analysis:
            total_regions = sum(len(p.get("detections", [])) for p in layout_data.get("pages", []))
            analysis["spatial_confidence"] = min(0.9, 0.5 + (total_regions * 0.05))
        
        if "hierarchy_analysis" not in analysis:
            analysis["hierarchy_analysis"] = self._create_hierarchy_analysis(layout_data)
        else:
            # Ensure hierarchy_analysis values are properly quoted
            analysis["hierarchy_analysis"] = self._fix_hierarchy_analysis(analysis["hierarchy_analysis"])
        
        if "region_importance_scores" not in analysis:
            analysis["region_importance_scores"] = self._create_importance_scores(layout_data)
        
        if "visual_validation_notes" not in analysis:
            analysis["visual_validation_notes"] = ["Layout analysis completed with LLM"]
        
        # Clean all strings recursively
        analysis = self._clean_structure(analysis)
        
        return analysis
    
    def _fix_hierarchy_analysis(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """Fix hierarchy analysis to ensure all values are properly quoted."""
        def fix_hierarchy_value(value):
            if isinstance(value, dict):
                return {k: fix_hierarchy_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [fix_hierarchy_value(item) for item in value]
            elif isinstance(value, str):
                # Already a string, return as is
                return value
            elif value is None:
                return None
            elif isinstance(value, (int, float, bool)):
                return value
            else:
                # Convert to string (this fixes p3_r0, p2_r0, etc.)
                return str(value)
        
        return fix_hierarchy_value(hierarchy)
    
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
    
    def _infer_reading_order(self, layout_data: Dict[str, Any]) -> List[str]:
        """Infer reading order based on positions."""
        reading_order = []
        
        if "pages" in layout_data:
            for page in layout_data["pages"]:
                page_num = page.get("page_number", 1)
                detections = page.get("detections", [])
                
                def get_y_pos(det):
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    if isinstance(bbox, list) and len(bbox) >= 2:
                        return bbox[1]
                    elif isinstance(bbox, dict):
                        return bbox.get("y1", 0)
                    return 0
                
                def get_x_pos(det):
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    if isinstance(bbox, list) and len(bbox) >= 1:
                        return bbox[0]
                    elif isinstance(bbox, dict):
                        return bbox.get("x1", 0)
                    return 0
                
                sorted_detections = sorted(detections, key=lambda d: (get_y_pos(d), get_x_pos(d)))
                
                for i, det in enumerate(sorted_detections):
                    region_id = f"p{page_num}_r{i}"
                    reading_order.append(region_id)
        
        return reading_order
    
    def _create_hierarchy_analysis(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic hierarchy analysis."""
        hierarchy = {
            "document_structure": "multi-page" if len(layout_data.get("pages", [])) > 1 else "single-page",
            "total_pages": len(layout_data.get("pages", [])),
            "regions_per_page": {}
        }
        
        for page in layout_data.get("pages", []):
            page_num = page.get("page_number", 1)
            hierarchy["regions_per_page"][f"page_{page_num}"] = len(page.get("detections", []))
        
        return hierarchy
    
    def _create_importance_scores(self, layout_data: Dict[str, Any]) -> Dict[str, float]:
        """Create basic importance scores."""
        scores = {}
        
        for page in layout_data.get("pages", []):
            page_num = page.get("page_number", 1)
            detections = page.get("detections", [])
            
            for i, det in enumerate(detections):
                region_id = f"p{page_num}_r{i}"
                confidence = det.get("confidence", 0.5)
                bbox = det.get("bbox", [0, 0, 100, 100])
                
                if isinstance(bbox, list) and len(bbox) >= 4:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                elif isinstance(bbox, dict):
                    area = (bbox.get("x2", 100) - bbox.get("x1", 0)) * (bbox.get("y2", 100) - bbox.get("y1", 0))
                else:
                    area = 1000
                
                normalized_area = min(1.0, area / 500000.0)
                importance = (confidence * 0.6) + (normalized_area * 0.4)
                scores[region_id] = round(importance, 3)
        
        return scores
    
    def _create_basic_analysis(self, layout_data: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        """Create basic analysis when LLM fails."""
        logger.info("Creating basic vision analysis (LLM unavailable)")
        
        return {
            "agent": self.name,
            "document_id": document_id,
            "logical_reading_order": self._infer_reading_order(layout_data),
            "hierarchy_analysis": self._create_hierarchy_analysis(layout_data),
            "region_importance_scores": self._create_importance_scores(layout_data),
            "visual_validation_notes": ["Basic layout analysis - LLM unavailable or failed"],
            "spatial_confidence": 0.5,
            "fallback_mode": True
        }
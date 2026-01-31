"""
JSON Validator Utility - Used by ALL agents to ensure valid JSON.
"""
import json
import re
import logging
import ast
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class JSONValidator:
    """Validates and fixes JSON for all agents."""
    
    @staticmethod
    def validate_and_fix(json_str: str) -> str:
        """Validate and fix JSON string with aggressive fixing."""
        if not json_str or not json_str.strip():
            return "{}"
        
        json_str = json_str.strip()
        
        # Step 1: Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        json_str = re.sub(r'```', '', json_str)
        
        # Step 2: Remove control characters (THIS FIXES "Invalid control character" error)
        json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
        
        # Step 3: Extract JSON object
        json_match = re.search(r'(\{.*\})', json_str, re.DOTALL)
        if not json_match:
            # Try to find start and end
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = json_str[start:end+1]
            else:
                return "{}"
        else:
            json_str = json_match.group(1)
        
        # Step 4: Apply critical fixes
        json_str = JSONValidator._apply_critical_fixes(json_str)
        
        # Step 5: Validate and try to parse
        for attempt in range(3):
            try:
                parsed = json.loads(json_str)
                # Success! Re-serialize to ensure it's clean
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse attempt {attempt+1} failed: {e.msg}")
                json_str = JSONValidator._advanced_fix(json_str, e.pos)
        
        # If all attempts fail, use ast.literal_eval as last resort
        try:
            import ast
            # Convert to Python dict
            python_str = json_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            parsed = ast.literal_eval(python_str)
            
            # Convert back to proper JSON
            def convert(obj):
                if isinstance(obj, dict):
                    return {str(k): convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(item) for item in obj]
                elif obj is None:
                    return None
                elif isinstance(obj, bool):
                    return obj
                elif isinstance(obj, (int, float)):
                    return obj
                else:
                    return str(obj)
            
            parsed = convert(parsed)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            logger.error(f"All JSON fixing failed: {e}")
            return "{}"
    
    @staticmethod
    def _apply_critical_fixes(json_str: str) -> str:
        """Apply critical JSON fixes that cause RAG failures."""
        # FIX 1: Quote unquoted string values (like "parent": p3_r0)
        # This is the MAIN FIX for the recurring error
        def fix_unquoted_strings(match):
            key = match.group(1)
            value = match.group(2)
            
            # Check what type of value this is
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                # Already quoted
                if value.startswith("'"):
                    value = f'"{value[1:-1]}"'
                return f'"{key}": {value}'
            elif re.match(r'^-?\d+(\.\d+)?$', value):
                # Number
                return f'"{key}": {value}'
            elif value.lower() in ['true', 'false', 'null']:
                # Boolean or null
                return f'"{key}": {value.lower()}'
            elif value.startswith('[') or value.startswith('{'):
                # Array or object
                return f'"{key}": {value}'
            else:
                # String value - quote it (THIS FIXES "parent": p3_r0)
                # Clean and escape
                value = value.replace('"', '\\"').replace("'", "\\'")
                return f'"{key}": "{value}"'
        
        # Pattern for key: value where value might be unquoted
        pattern = r'"([^"]+)"\s*:\s*([^,\[\]{}\s"\'][^,\[\]{}"\']*)(?=[,\]}])'
        json_str = re.sub(pattern, fix_unquoted_strings, json_str)
        
        # FIX 2: Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # FIX 3: Escape quotes inside strings
        def escape_inner_quotes(match):
            content = match.group(1)
            # Escape unescaped quotes
            content = re.sub(r'(?<!\\)"', r'\"', content)
            return f'"{content}"'
        
        json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_inner_quotes, json_str)
        
        # FIX 4: Fix property names
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # FIX 5: Fix arrays with unquoted strings
        def fix_array(match):
            array_content = match.group(1)
            parts = re.split(r',\s*(?![^\[\]]*\])', array_content)
            fixed_parts = []
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                if (part.startswith('"') and part.endswith('"')) or \
                   re.match(r'^-?\d+(\.\d+)?$', part) or \
                   part in ['true', 'false', 'null'] or \
                   part.startswith('[') or part.startswith('{'):
                    fixed_parts.append(part)
                else:
                    # Quote unquoted strings in arrays
                    fixed_parts.append(f'"{part}"')
            
            return '[' + ', '.join(fixed_parts) + ']'
        
        json_str = re.sub(r'\[(.*?)\]', fix_array, json_str, flags=re.DOTALL)
        
        return json_str
    
    @staticmethod
    def _advanced_fix(json_str: str, error_pos: int) -> str:
        """Advanced JSON fixing based on error position."""
        if error_pos >= len(json_str):
            return json_str
        
        # Get context around error
        start = max(0, error_pos - 30)
        end = min(len(json_str), error_pos + 30)
        context = json_str[start:end]
        
        logger.debug(f"Error context: ...{context}...")
        
        # Fix common issues at error position
        if error_pos < len(json_str):
            char_at_pos = json_str[error_pos]
            
            # Fix unescaped quotes
            if char_at_pos == '"':
                # Look for unescaped quote
                if error_pos > 0 and json_str[error_pos-1] != '\\':
                    # Escape it
                    json_str = json_str[:error_pos] + '\\"' + json_str[error_pos+1:]
            
            # Fix missing comma
            elif char_at_pos == '"' and error_pos > 0:
                # Check if missing comma before this
                prev_char = json_str[error_pos-1]
                if prev_char in ['}', '"']:
                    json_str = json_str[:error_pos] + ', ' + json_str[error_pos:]
        
        return json_str
    
    @staticmethod
    def ensure_valid_structure(data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure data structure is valid JSON."""
        def clean_value(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    # Ensure key is string
                    k_str = str(k)
                    # Clean value
                    cleaned[k_str] = clean_value(v)
                return cleaned
            elif isinstance(obj, list):
                return [clean_value(item) for item in obj]
            elif isinstance(obj, str):
                # Clean string
                obj = obj.replace('\n', ' ').replace('\r', ' ')
                obj = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', obj)
                obj = obj.strip()
                return obj
            elif isinstance(obj, (int, float, bool)):
                return obj
            elif obj is None:
                return None
            else:
                return str(obj)
        
        return clean_value(data)


# Global instance
json_validator = JSONValidator()
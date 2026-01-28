# app/llm/client.py - COMPLETE FIXED VERSION
"""
Open-source LLM client for Qwen-2.5-Instruct.
Supports Ollama, vLLM, or any OpenAI-compatible API.
"""
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for open-source LLM inference."""
    
    def __init__(self, base_url: str = None, model: str = "qwen2.5:1.5b"):
        """
        Initialize LLM client.
        
        Args:
            base_url: Base URL for OpenAI-compatible API (e.g., http://localhost:11434/v1 for Ollama)
            model: Model name to use
        """
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:1.5b")
        
        # Initialize OpenAI client with open-source endpoint
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="ollama" if "ollama" in self.base_url else "not-needed"
        )
        
        logger.info(f"LLM Client initialized: {self.base_url}, model={self.model}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.1) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.info(f"📡 LLM Request - Model: {self.model}, Prompt length: {len(prompt)}")
            
            # Add timeout
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=4000
                    ),
                    timeout=60.0  # 60 second timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM request timed out after 60 seconds")
                raise Exception("LLM request timed out")
            
            if not response or not response.choices:
                logger.error("LLM returned empty response")
                raise Exception("Empty response from LLM")
            
            result = response.choices[0].message.content
            
            if not result or len(result.strip()) == 0:
                logger.error("LLM returned empty content")
                raise Exception("Empty content from LLM")
            
            logger.info(f"✅ LLM generated {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            # RE-RAISE the exception instead of returning empty string
            raise Exception(f"LLM failed: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_structured(self, prompt: str, system_prompt: Optional[str] = None,
                                json_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_schema: Optional JSON schema for response
            
        Returns:
            Structured dictionary
        """
        full_prompt = prompt
        if json_schema:
            schema_str = str(json_schema)
            full_prompt += f"\n\nReturn valid JSON matching this schema:\n{schema_str}"
        
        response = await self.generate(full_prompt, system_prompt, temperature=0.1)
        
        # Try to parse JSON from response
        try:
            import json
            import re
            
            # Extract JSON from markdown if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Clean the response
            response = response.strip()
            
            # Try to find JSON object or array
            json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                logger.info(f"✅ Successfully parsed JSON from LLM response")
                return result
            else:
                # Try to parse entire response as JSON
                result = json.loads(response)
                return result
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw response: {response[:200]}...")
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response[:500]
            }


# Global instance
llm_client = LLMClient()
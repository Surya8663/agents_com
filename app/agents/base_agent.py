"""
Base agent class for all intelligence agents.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.llm.client import llm_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str):
        """Initialize agent with name."""
        self.name = name
        self.llm = llm_client
        logger.info(f"Initialized agent: {name}")
    
    @abstractmethod
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and return outputs.
        
        Args:
            inputs: Agent-specific inputs
            
        Returns:
            Agent outputs
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that required input keys are present."""
        for key in required_keys:
            if key not in inputs:
                logger.error(f"Missing required input key: {key}")
                return False
        return True
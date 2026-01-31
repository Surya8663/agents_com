"""
LangGraph orchestration for multi-agent workflow - REAL DATA ONLY.
"""
import logging
import time
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

from app.agents.vision_agent import VisionAgent
from app.agents.text_agent import TextAgent
from app.agents.fusion_agent import FusionAgent
from app.agents.validation_agent import ValidationAgent

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for agent workflow."""
    document_id: str
    layout_data: Dict[str, Any]
    ocr_data: Dict[str, Any]
    vision_analysis: Dict[str, Any]
    text_analysis: Dict[str, Any]
    fused_document: Dict[str, Any]
    validation_result: Dict[str, Any]
    errors: List[str]
    current_agent: str
    start_time: float  # ADDED: Track start time
    agent_results: Dict[str, Dict[str, Any]]  # ADDED: Store all agent results


class AgentOrchestrator:
    """Orchestrates multi-agent workflow using LangGraph."""
    
    def __init__(self):
        """Initialize orchestrator with agents."""
        self.vision_agent = VisionAgent()
        self.text_agent = TextAgent()
        self.fusion_agent = FusionAgent()
        self.validation_agent = ValidationAgent()
        
        # Build workflow graph WITHOUT checkpointing to avoid timestamp issues
        self.workflow = self._build_workflow()
        logger.info("✅ Agent workflow graph built")
    
    def _build_workflow(self):
        """Build LangGraph workflow WITHOUT checkpointing."""
        # Define nodes
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("vision_agent", self._run_vision_agent)
        workflow.add_node("text_agent", self._run_text_agent)
        workflow.add_node("fusion_agent", self._run_fusion_agent)
        workflow.add_node("validation_agent", self._run_validation_agent)
        
        # Add edges
        workflow.add_edge("vision_agent", "text_agent")
        workflow.add_edge("text_agent", "fusion_agent")
        workflow.add_edge("fusion_agent", "validation_agent")
        workflow.add_edge("validation_agent", END)
        
        # Set entry point
        workflow.set_entry_point("vision_agent")
        
        # IMPORTANT: Don't use MemorySaver to avoid timestamp issues
        # Just compile without checkpointing
        return workflow.compile()
    
    async def _run_vision_agent(self, state: AgentState) -> AgentState:
        """Run Vision Agent."""
        try:
            logger.info("👁️ Running Vision Agent...")
            
            inputs = {
                "document_id": state["document_id"],
                "layout_data": state["layout_data"]
            }
            
            result = await self.vision_agent.process(inputs)
            
            logger.info(f"✅ Vision Agent completed - Confidence: {result.get('spatial_confidence', 0.0):.2f}")
            return {
                **state,
                "vision_analysis": result,
                "current_agent": "vision_agent",
                "agent_results": {**state.get("agent_results", {}), "vision": result}
            }
            
        except Exception as e:
            logger.error(f"❌ Vision Agent failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Vision Agent: {str(e)}"],
                "vision_analysis": {"error": str(e), "agent": "vision_agent", "fallback_mode": True},
                "current_agent": "vision_agent",
                "agent_results": {**state.get("agent_results", {}), "vision": {"error": str(e)}}
            }
    
    async def _run_text_agent(self, state: AgentState) -> AgentState:
        """Run Text Agent."""
        try:
            logger.info("🔤 Running Text Agent...")
            
            inputs = {
                "document_id": state["document_id"],
                "ocr_data": state["ocr_data"]
            }
            
            result = await self.text_agent.process(inputs)
            
            doc_type = result.get('document_type', 'unknown')
            logger.info(f"✅ Text Agent completed - Document type: {doc_type}")
            logger.info(f"   Semantic confidence: {result.get('semantic_confidence', 0.0):.2f}")
            
            return {
                **state,
                "text_analysis": result,
                "current_agent": "text_agent",
                "agent_results": {**state.get("agent_results", {}), "text": result}
            }
            
        except Exception as e:
            logger.error(f"❌ Text Agent failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Text Agent: {str(e)}"],
                "text_analysis": {"error": str(e), "agent": "text_agent", "fallback_mode": True},
                "current_agent": "text_agent",
                "agent_results": {**state.get("agent_results", {}), "text": {"error": str(e)}}
            }
    
    async def _run_fusion_agent(self, state: AgentState) -> AgentState:
        """Run Fusion Agent."""
        try:
            logger.info("🤝 Running Fusion Agent...")
            
            # Ensure we have the required inputs
            if "vision_analysis" not in state or not state["vision_analysis"]:
                state["vision_analysis"] = {"error": "Missing vision analysis", "fallback_mode": True}
            if "text_analysis" not in state or not state["text_analysis"]:
                state["text_analysis"] = {"error": "Missing text analysis", "fallback_mode": True}
            
            inputs = {
                "document_id": state["document_id"],
                "vision_analysis": state["vision_analysis"],
                "text_analysis": state["text_analysis"],
                "layout_data": state.get("layout_data", {}),
                "ocr_data": state.get("ocr_data", {})
            }
            
            result = await self.fusion_agent.process(inputs)
            
            fusion_conf = result.get('fusion_confidence', 0.0)
            logger.info(f"✅ Fusion Agent completed - Confidence: {fusion_conf:.2f}")
            
            return {
                **state,
                "fused_document": result,
                "current_agent": "fusion_agent",
                "agent_results": {**state.get("agent_results", {}), "fusion": result}
            }
            
        except Exception as e:
            logger.error(f"❌ Fusion Agent failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Fusion Agent: {str(e)}"],
                "fused_document": {"error": str(e), "agent": "fusion_agent", "fallback_mode": True},
                "current_agent": "fusion_agent",
                "agent_results": {**state.get("agent_results", {}), "fusion": {"error": str(e)}}
            }
    
    async def _run_validation_agent(self, state: AgentState) -> AgentState:
        """Run Validation Agent."""
        try:
            logger.info("✅ Running Validation Agent...")
            
            # Ensure we have required inputs
            if "fused_document" not in state or not state["fused_document"]:
                state["fused_document"] = {"error": "Missing fused document", "fallback_mode": True}
            
            inputs = {
                "document_id": state["document_id"],
                "fused_document": state["fused_document"],
                "vision_analysis": state.get("vision_analysis", {}),
                "text_analysis": state.get("text_analysis", {})
            }
            
            result = await self.validation_agent.process(inputs)
            
            overall_conf = result.get('overall_confidence', 0.0)
            logger.info(f"✅ Validation Agent completed - Overall confidence: {overall_conf:.2f}")
            
            return {
                **state,
                "validation_result": result,
                "current_agent": "validation_agent",
                "agent_results": {**state.get("agent_results", {}), "validation": result}
            }
            
        except Exception as e:
            logger.error(f"❌ Validation Agent failed: {e}")
            return {
                **state,
                "errors": state.get("errors", []) + [f"Validation Agent: {str(e)}"],
                "validation_result": {"error": str(e), "agent": "validation_agent", "fallback_mode": True},
                "current_agent": "validation_agent",
                "agent_results": {**state.get("agent_results", {}), "validation": {"error": str(e)}}
            }
    
    async def run(self, document_id: str, layout_data: Dict[str, Any], 
                 ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete agent workflow.
        
        Args:
            document_id: Document identifier
            layout_data: Phase 2 layout data
            ocr_data: Phase 3 OCR data
            
        Returns:
            Complete agent results
        """
        logger.info(f"🚀 Starting agent workflow for document {document_id}")
        
        # Initial state
        initial_state = AgentState(
            document_id=document_id,
            layout_data=layout_data,
            ocr_data=ocr_data,
            vision_analysis={},
            text_analysis={},
            fused_document={},
            validation_result={},
            errors=[],
            current_agent="start",
            start_time=time.time(),
            agent_results={}
        )
        
        try:
            # Execute workflow WITHOUT checkpoint config
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Compile final result
            result = self._compile_final_result(final_state)
            
            elapsed_time = time.time() - initial_state["start_time"]
            logger.info(f"🎉 Agent workflow completed for {document_id} in {elapsed_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Agent workflow failed: {e}")
            return self._create_error_result(document_id, str(e))
    
    def _compile_final_result(self, state: AgentState) -> Dict[str, Any]:
        """Compile final result from workflow state."""
        # Determine status
        errors = state.get("errors", [])
        if errors:
            status = "completed_with_errors"
            logger.warning(f"Workflow completed with {len(errors)} errors")
        else:
            status = "completed"
        
        # Get which agents actually executed
        agents_executed = []
        vision_result = state.get("vision_analysis", {})
        text_result = state.get("text_analysis", {})
        fusion_result = state.get("fused_document", {})
        validation_result = state.get("validation_result", {})
        
        if vision_result and not vision_result.get("error"):
            agents_executed.append("vision_agent")
        if text_result and not text_result.get("error"):
            agents_executed.append("text_agent")
        if fusion_result and not fusion_result.get("error"):
            agents_executed.append("fusion_agent")
        if validation_result and not validation_result.get("error"):
            agents_executed.append("validation_agent")
        
        # Create final output
        final_output = self._create_final_output(state)
        
        return {
            "document_id": state["document_id"],
            "status": status,
            "agents_executed": agents_executed,
            "vision_analysis": vision_result,
            "text_analysis": text_result,
            "fused_document": fusion_result,
            "validation_result": validation_result,
            "errors": errors,
            "final_output": final_output,
            "agent_results": state.get("agent_results", {}),
            "timestamp": datetime.now().isoformat() if 'datetime' in globals() else time.ctime()
        }
    
    def _create_final_output(self, state: AgentState) -> Dict[str, Any]:
        """Create final structured output."""
        fused_doc = state.get("fused_document", {})
        validation = state.get("validation_result", {})
        text_analysis = state.get("text_analysis", {})
        
        # EXTRACT DATA FROM MULTIPLE SOURCES
        extracted_fields = {}
        
        # 1. Try fused_extractions first
        fused_extractions = fused_doc.get("fused_extractions", {})
        if fused_extractions and isinstance(fused_extractions, dict):
            logger.info(f"📊 Found {len(fused_extractions)} fused extractions")
            for key, value_info in fused_extractions.items():
                if isinstance(value_info, dict):
                    # Format: {"key": {"value": "actual", "multi_modal_confidence": 0.9}}
                    extracted_fields[key] = {
                        "value": value_info.get("value", str(value_info)),
                        "confidence": value_info.get("multi_modal_confidence", 
                                                    fused_doc.get("fusion_confidence", 0.0)),
                        "source": "multi_modal_fusion"
                    }
                else:
                    # Format: {"key": "actual_value"}
                    extracted_fields[key] = {
                        "value": str(value_info),
                        "confidence": fused_doc.get("fusion_confidence", 0.0),
                        "source": "fusion_agent"
                    }
        
        # 2. Fallback to text analysis key_value_pairs
        if not extracted_fields:
            key_value_pairs = text_analysis.get("key_value_pairs", {})
            if key_value_pairs and isinstance(key_value_pairs, dict):
                logger.info(f"📊 Using {len(key_value_pairs)} text analysis extractions")
                for key, value in key_value_pairs.items():
                    extracted_fields[key] = {
                        "value": str(value),
                        "confidence": text_analysis.get("semantic_confidence", 0.0),
                        "source": "text_agent"
                    }
        
        # 3. Fallback to text analysis key_entities
        if not extracted_fields:
            key_entities = text_analysis.get("key_entities", {})
            if key_entities and isinstance(key_entities, dict):
                for entity_type, entities in key_entities.items():
                    if entities and isinstance(entities, list) and entities:
                        extracted_fields[entity_type] = {
                            "value": str(entities[0]),
                            "confidence": text_analysis.get("semantic_confidence", 0.0),
                            "source": "text_agent_entities"
                        }
        
        # Handle validation notes format
        validation_notes = []
        explainability_notes = validation.get("explainability_notes", [])
        
        if isinstance(explainability_notes, dict):
            # Convert dict to list
            for key, value in explainability_notes.items():
                validation_notes.append(f"{key}: {value}")
        elif isinstance(explainability_notes, list):
            validation_notes = explainability_notes
        elif explainability_notes:
            validation_notes = [str(explainability_notes)]
        
        # Add validation passed/failed notes
        passed = validation.get("validation_passed", [])
        failed = validation.get("validation_failed", [])
        
        if passed:
            validation_notes.append(f"Passed checks: {', '.join(passed[:3])}" + 
                                  (f" and {len(passed)-3} more" if len(passed) > 3 else ""))
        if failed:
            validation_notes.append(f"Failed checks: {', '.join(failed[:3])}" + 
                                  (f" and {len(failed)-3} more" if len(failed) > 3 else ""))
        
        # Get document type from multiple sources
        document_type = "unknown"
        if fused_doc.get("unified_structure", {}).get("document_type"):
            document_type = fused_doc["unified_structure"]["document_type"]
        elif text_analysis.get("document_type"):
            document_type = text_analysis["document_type"]
        elif fused_doc.get("document_type"):
            document_type = fused_doc["document_type"]
        
        # Log what we extracted
        if extracted_fields:
            logger.info(f"🎯 Final output: Extracted {len(extracted_fields)} fields")
            for key, info in list(extracted_fields.items())[:5]:
                logger.info(f"   • {key}: {info.get('value', 'N/A')}")
        else:
            logger.warning("⚠️ No fields extracted in final output")
        
        return {
            "document_id": state["document_id"],
            "extracted_fields": extracted_fields,
            "confidence_score": validation.get("overall_confidence", 0.0),
            "validation_notes": validation_notes,
            "traceability": {
                "vision_regions": self._extract_vision_regions(state.get("layout_data", {})),
                "ocr_text_sources": self._extract_ocr_sources(state.get("ocr_data", {})),
                "agent_confidence_scores": {
                    "vision": state.get("vision_analysis", {}).get("spatial_confidence", 0.0),
                    "text": state.get("text_analysis", {}).get("semantic_confidence", 0.0),
                    "fusion": fused_doc.get("fusion_confidence", 0.0),
                    "validation": validation.get("overall_confidence", 0.0)
                }
            },
            "document_type": document_type,
            "processing_errors": state.get("errors", []),
            "timestamp": datetime.now().isoformat() if 'datetime' in globals() else time.ctime()
        }
    
    def _extract_vision_regions(self, layout_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract vision regions for traceability."""
        regions = []
        
        if "pages" in layout_data:
            for page in layout_data["pages"]:
                page_num = page.get("page_number", 1)
                detections = page.get("detections", [])
                
                for i, det in enumerate(detections):
                    regions.append({
                        "region_id": f"p{page_num}_r{i}",
                        "type": det.get("type", det.get("label", "unknown")),
                        "bbox": det.get("bbox", {}),
                        "confidence": det.get("confidence", 0.0),
                        "page": page_num
                    })
        
        return regions[:20]  # Limit to 20 regions
    
    def _extract_ocr_sources(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract OCR sources for traceability."""
        sources = []
        
        if "pages" in ocr_data:
            for page in ocr_data["pages"]:
                page_num = page.get("page_number", 1)
                regions = page.get("regions", [])
                
                for i, region in enumerate(regions):
                    text = region.get("ocr_text", "")
                    if text and str(text).strip():
                        sources.append({
                            "region_id": f"p{page_num}_r{i}",
                            "text_preview": str(text)[:80] + "..." if len(str(text)) > 80 else str(text),
                            "confidence": region.get("ocr_confidence", 0.0),
                            "page": page_num,
                            "region_type": region.get("type", "unknown")
                        })
        
        return sources[:20]  # Limit to 20 sources
    
    def _create_error_result(self, document_id: str, error: str) -> Dict[str, Any]:
        """Create error result when workflow fails."""
        from datetime import datetime
        
        return {
            "document_id": document_id,
            "status": "failed",
            "error": error,
            "agents_executed": [],
            "final_output": {
                "document_id": document_id,
                "extracted_fields": {},
                "confidence_score": 0.0,
                "validation_notes": [f"Workflow failed: {error}"],
                "traceability": {},
                "document_type": "unknown",
                "processing_errors": [error],
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()
"""
Multi-modal indexer for RAG system.
"""
import logging
import uuid
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.rag.multimodal_schema import RAGChunk, Modality
from app.rag.embedding_manager import embedding_manager
from app.vectorstore.qdrant_client import qdrant_store
from app.config.settings import settings

logger = logging.getLogger(__name__)


class MultiModalIndexer:
    """Indexes multi-modal document chunks."""
    
    def __init__(self):
        """Initialize indexer."""
        self.embedding_manager = embedding_manager
    
    def index_document(self, document_id: str, agent_results: Optional[Dict[str, Any]] = None) -> bool:
        """Index an entire document from stored agent results."""
        try:
            logger.info(f"📥 Indexing document {document_id}")
            
            # Load agent results if not provided
            if agent_results is None:
                logger.info(f"Loading agent results for {document_id}")
                agent_results = self._load_agent_results(document_id)
                if not agent_results:
                    logger.error(f"❌ No agent results found for document {document_id}")
                    return False
            
            # Clean and validate agent results
            agent_results = self._clean_and_validate_agent_results(agent_results, document_id)
            
            # Create chunks
            chunks = self.create_chunks_from_agent_results(agent_results)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created for document {document_id}")
                # Still try to index minimal info
                chunks = self._create_minimal_chunks(agent_results, document_id)
            
            # Index chunks
            success = self.index_chunks(chunks)
            
            if success:
                logger.info(f"✅ Successfully indexed {len(chunks)} chunks for document {document_id}")
            else:
                logger.error(f"❌ Failed to index chunks for document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Document indexing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_agent_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Load agent results from file."""
        try:
            doc_dir = settings.DOCUMENTS_DIR / document_id
            
            # Try multiple possible locations
            possible_paths = [
                doc_dir / "agents" / "agent_results.json",
                doc_dir / "agents" / "cleaned_agent_results.json",
                doc_dir / "agent_results.json",
                doc_dir / "agents.json"
            ]
            
            for agent_file in possible_paths:
                if agent_file.exists():
                    logger.info(f"📄 Loading agent results from {agent_file}")
                    with open(agent_file, "r", encoding="utf-8", errors='ignore') as f:
                        content = f.read()
                    
                    # Clean the content before parsing
                    content = self._clean_json_content(content)
                    
                    try:
                        agent_results = json.loads(content)
                        logger.info(f"✅ Successfully loaded agent results from {agent_file}")
                        return agent_results
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ JSON decode error in {agent_file}: {e}")
                        # Try to repair
                        agent_results = self._repair_json(content)
                        if agent_results:
                            return agent_results
            
            logger.error(f"❌ No agent results file found for {document_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load agent results: {e}")
            return None
    
    def _clean_json_content(self, content: str) -> str:
        """Clean JSON content before parsing."""
        if not content:
            return "{}"
        
        # Remove null bytes and control characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # Fix common JSON issues
        content = content.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        
        # Fix unquoted string values (e.g., "parent": p3_r0)
        def quote_unquoted(match):
            key = match.group(1)
            value = match.group(2)
            
            # Don't quote if it's a number, boolean, null, or already quoted
            if re.match(r'^-?\d+(\.\d+)?$', value):  # Number
                return f'"{key}": {value}'
            elif value.lower() in ['true', 'false', 'null']:  # Boolean/null
                return f'"{key}": {value.lower()}'
            elif value.startswith('[') or value.startswith('{'):  # Array/object
                return f'"{key}": {value}'
            elif (value.startswith('"') and value.endswith('"')) or \
                 (value.startswith("'") and value.endswith("'")):  # Already quoted
                return f'"{key}": {value}'
            else:  # String - quote it
                return f'"{key}": "{value}"'
        
        # Pattern for key: unquoted_value
        content = re.sub(r'"([^"]+)"\s*:\s*([^,\[\]{}\s"\'][^,\[\]{}]*)(?=[,\]}])', 
                        quote_unquoted, content)
        
        # Fix trailing commas
        content = re.sub(r',\s*([}\]])', r'\1', content)
        
        # Ensure it's a proper JSON object
        if not content.startswith('{'):
            content = '{' + content
        if not content.endswith('}'):
            content = content + '}'
        
        return content
    
    def _repair_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair broken JSON."""
        try:
            # Try to parse as Python dict (more forgiving)
            import ast
            content = content.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            parsed = ast.literal_eval(content)
            
            # Convert back to dict
            def convert_value(v):
                if isinstance(v, dict):
                    return {k: convert_value(v2) for k, v2 in v.items()}
                elif isinstance(v, list):
                    return [convert_value(item) for item in v]
                elif v is None:
                    return None
                elif isinstance(v, (bool, int, float, str)):
                    return v
                else:
                    return str(v)
            
            return convert_value(parsed)
            
        except Exception as e:
            logger.warning(f"Could not repair JSON: {e}")
            
            # Extract what we can
            result = {}
            try:
                # Extract simple key-value pairs
                pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
                matches = re.findall(pattern, content)
                for key, value in matches:
                    result[key] = value
                
                if result:
                    logger.info(f"📊 Extracted {len(result)} fields from broken JSON")
                    return result
            except Exception as e2:
                logger.warning(f"Could not extract partial JSON: {e2}")
            
            return None
    
    def _clean_and_validate_agent_results(self, agent_results: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        """Clean and validate agent results structure."""
        if not isinstance(agent_results, dict):
            agent_results = {}
        
        # Ensure document_id
        agent_results["document_id"] = document_id
        
        # Clean all string values
        agent_results = self._clean_all_strings(agent_results)
        
        # Ensure basic structure
        if "final_output" not in agent_results:
            agent_results["final_output"] = {
                "document_id": document_id,
                "extracted_fields": {},
                "confidence_score": 0.0,
                "document_type": "unknown"
            }
        
        # Ensure agent analyses exist (even if empty)
        for agent_type in ["vision_analysis", "text_analysis", "fused_document"]:
            if agent_type not in agent_results:
                agent_results[agent_type] = {}
        
        return agent_results
    
    def _clean_all_strings(self, data: Any) -> Any:
        """Recursively clean all string values in the data."""
        if isinstance(data, dict):
            return {k: self._clean_all_strings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_all_strings(item) for item in data]
        elif isinstance(data, str):
            # Clean string
            data = data.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            data = re.sub(r'\s+', ' ', data).strip()
            # Remove problematic characters
            data = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', data)
            return data
        else:
            return data
    
    def create_chunks_from_agent_results(self, agent_results: Dict[str, Any]) -> List[RAGChunk]:
        """Create RAG chunks from agent pipeline results."""
        chunks = []
        document_id = agent_results.get("document_id", "unknown")
        
        try:
            # Extract from fused document
            fused_doc = agent_results.get("fused_document", {})
            text_analysis = agent_results.get("text_analysis", {})
            vision_analysis = agent_results.get("vision_analysis", {})
            layout_data = agent_results.get("layout_data", {})
            ocr_data = agent_results.get("ocr_data", {})
            final_output = agent_results.get("final_output", {})
            
            logger.info(f"Creating chunks from agent results for {document_id}")
            
            # 1. Text chunks from key-value pairs
            key_value_pairs = text_analysis.get("key_value_pairs", {})
            if key_value_pairs and isinstance(key_value_pairs, dict):
                for key, value in key_value_pairs.items():
                    if isinstance(value, dict):
                        value_str = value.get("value", str(value))
                    else:
                        value_str = str(value)
                    
                    if value_str and len(value_str.strip()) > 0:
                        chunk = RAGChunk(
                            id=f"text_{uuid.uuid4().hex[:8]}",
                            document_id=document_id,
                            page_number=1,
                            text=f"{key}: {value_str}",
                            modality=Modality.TEXT,
                            agent_source="text_agent",
                            confidence=float(text_analysis.get("semantic_confidence", 0.7)),
                            metadata={"field_type": "key_value_pair", "key": key}
                        )
                        chunks.append(chunk)
            
            # 2. Text chunks from final_output extracted_fields
            extracted_fields = final_output.get("extracted_fields", {})
            if extracted_fields and isinstance(extracted_fields, dict):
                for key, field_info in extracted_fields.items():
                    if isinstance(field_info, dict):
                        value = field_info.get("value", "")
                        confidence = field_info.get("confidence", 0.7)
                    else:
                        value = str(field_info)
                        confidence = final_output.get("confidence_score", 0.7)
                    
                    if value and len(str(value).strip()) > 0:
                        chunk = RAGChunk(
                            id=f"extract_{uuid.uuid4().hex[:8]}",
                            document_id=document_id,
                            page_number=1,
                            text=f"{key}: {value}",
                            modality=Modality.TEXT,
                            agent_source="final_output",
                            confidence=float(confidence),
                            metadata={"field_type": "extracted_field", "key": key}
                        )
                        chunks.append(chunk)
            
            # 3. Text chunks from OCR regions
            if "pages" in ocr_data and isinstance(ocr_data["pages"], list):
                for page in ocr_data["pages"]:
                    page_num = page.get("page_number", 1)
                    regions = page.get("regions", [])
                    if isinstance(regions, list):
                        for region in regions:
                            if isinstance(region, dict):
                                ocr_text = region.get("ocr_text", "")
                                if ocr_text and len(str(ocr_text).strip()) > 0:
                                    chunk = RAGChunk(
                                        id=f"ocr_{uuid.uuid4().hex[:8]}",
                                        document_id=document_id,
                                        page_number=page_num,
                                        text=str(ocr_text),
                                        modality=Modality.TEXT,
                                        agent_source="ocr_engine",
                                        confidence=float(region.get("ocr_confidence", 0.5)),
                                        metadata={
                                            "region_type": region.get("type", "text_block"),
                                            "bbox": region.get("bbox", {}),
                                            "engine": "easyocr"
                                        }
                                    )
                                    chunks.append(chunk)
            
            # 4. Vision chunks from layout analysis
            if "pages" in layout_data and isinstance(layout_data["pages"], list):
                for page in layout_data["pages"]:
                    if isinstance(page, dict):
                        page_num = page.get("page_number", 1)
                        detections = page.get("detections", [])
                        if isinstance(detections, list):
                            for detection in detections:
                                if isinstance(detection, dict):
                                    chunk = RAGChunk(
                                        id=f"vision_{uuid.uuid4().hex[:8]}",
                                        document_id=document_id,
                                        page_number=page_num,
                                        visual_description=f"{detection.get('type', 'element')} at position {detection.get('bbox', {})}",
                                        bbox=detection.get("bbox"),
                                        region_type=detection.get("type", "unknown"),
                                        modality=Modality.VISION,
                                        agent_source="vision_agent",
                                        confidence=float(detection.get("confidence", 0.6)),
                                        metadata={
                                            "detection_type": detection.get("type"),
                                            "page_dimensions": f"{page.get('page_width', 0)}x{page.get('page_height', 0)}"
                                        }
                                    )
                                    chunks.append(chunk)
            
            # 5. Fusion chunks from fused extractions
            fused_extractions = fused_doc.get("fused_extractions", {})
            if fused_extractions and isinstance(fused_extractions, dict):
                for key, value_info in fused_extractions.items():
                    if isinstance(value_info, dict):
                        value = value_info.get("value", "")
                        confidence = value_info.get("multi_modal_confidence", 0.8)
                    else:
                        value = str(value_info)
                        confidence = fused_doc.get("fusion_confidence", 0.8)
                    
                    if value and len(str(value).strip()) > 0:
                        chunk = RAGChunk(
                            id=f"fusion_{uuid.uuid4().hex[:8]}",
                            document_id=document_id,
                            page_number=1,
                            text=f"{key}: {value}",
                            modality=Modality.MULTIMODAL,
                            agent_source="fusion_agent",
                            confidence=float(confidence),
                            metadata={
                                "extraction_type": "fused",
                                "source": "multi_modal_fusion"
                            }
                        )
                        chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} RAG chunks from document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error creating chunks: {e}")
            # Return minimal chunks if creation fails
            return self._create_minimal_chunks(agent_results, document_id)
    
    def _create_minimal_chunks(self, agent_results: Dict[str, Any], document_id: str) -> List[RAGChunk]:
        """Create minimal chunks when main creation fails."""
        chunks = []
        
        try:
            # Create at least one chunk with document info
            final_output = agent_results.get("final_output", {})
            doc_type = final_output.get("document_type", "unknown")
            confidence = final_output.get("confidence_score", 0.5)
            
            chunk = RAGChunk(
                id=f"minimal_{uuid.uuid4().hex[:8]}",
                document_id=document_id,
                page_number=1,
                text=f"Document: {doc_type}",
                modality=Modality.TEXT,
                agent_source="fallback",
                confidence=float(confidence),
                metadata={"fallback": True, "document_type": doc_type}
            )
            chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} minimal chunks for {document_id}")
            
        except Exception as e:
            logger.error(f"Even minimal chunk creation failed: {e}")
        
        return chunks
    
    def index_chunks(self, chunks: List[RAGChunk]) -> bool:
        """Index chunks with embeddings in vector store."""
        if not chunks:
            logger.warning("No chunks to index")
            return False
        
        try:
            document_id = chunks[0].document_id if chunks else ""
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                try:
                    # Generate text embedding
                    if chunk.text:
                        chunk.text_embedding = self.embedding_manager.embed_text(chunk.text)
                    else:
                        chunk.text_embedding = [0.0] * 384
                    
                    # Generate visual embedding
                    if chunk.visual_description:
                        chunk.visual_embedding = self.embedding_manager.embed_text(chunk.visual_description)
                    else:
                        chunk.visual_embedding = chunk.text_embedding
                        
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk {chunk.id}: {e}")
                    # Use zero vectors as fallback
                    chunk.text_embedding = [0.0] * 384
                    chunk.visual_embedding = [0.0] * 384
            
            # Prepare chunks for storage
            chunk_dicts = []
            for chunk in chunks:
                try:
                    chunk_dict = chunk.dict()
                    
                    # Use text embedding as primary vector
                    primary_embedding = chunk.text_embedding or [0.0] * 384
                    
                    # Ensure all values are JSON serializable
                    metadata = {}
                    for k, v in chunk.metadata.items():
                        if isinstance(v, (str, int, float, bool, type(None))):
                            metadata[k] = v
                        else:
                            metadata[k] = str(v)
                    
                    chunk_dicts.append({
                        "embedding": primary_embedding,
                        "text": chunk.text or "",
                        "visual_description": chunk.visual_description or "",
                        "metadata": {
                            "document_id": chunk.document_id,
                            "page_number": chunk.page_number,
                            "modality": chunk.modality,
                            "agent_source": chunk.agent_source,
                            "confidence": float(chunk.confidence) if chunk.confidence is not None else 0.0,
                            "region_type": chunk.region_type or "",
                            "bbox": chunk.bbox.dict() if chunk.bbox and hasattr(chunk.bbox, 'dict') else {},
                            **metadata
                        }
                    })
                except Exception as e:
                    logger.warning(f"Failed to prepare chunk {chunk.id} for storage: {e}")
                    continue
            
            if not chunk_dicts:
                logger.error("No chunks prepared for storage")
                return False
            
            # Store in vector database
            logger.info(f"Storing {len(chunk_dicts)} chunks in vector database")
            success = qdrant_store.store_document_chunks(document_id, chunk_dicts)
            
            if success:
                logger.info(f"✅ Indexed {len(chunk_dicts)} chunks for document {document_id}")
                return True
            else:
                logger.error(f"❌ Failed to index chunks for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# Global instance
indexer = MultiModalIndexer()
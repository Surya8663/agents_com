"""
Multi-modal retriever for RAG system.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid
from app.rag.multimodal_schema import RAGChunk, RAGQuery, RAGResult, RAGResponse, Modality
from app.rag.embedding_manager import embedding_manager
from app.vectorstore.qdrant_client import qdrant_store
from app.llm.client import llm_client

logger = logging.getLogger(__name__)


class MultiModalRetriever:
    """Retrieves multi-modal evidence for queries."""
    
    def __init__(self):
        """Initialize retriever."""
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: RAGQuery) -> List[RAGResult]:
        """Retrieve relevant chunks for a query."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_text(query.query)
            
            # Search in vector store
            raw_results = qdrant_store.search_similar(
                query_embedding=query_embedding,
                document_id=query.document_id,
                limit=query.top_k * 2  # Get extra for filtering
            )
            
            # Filter and score results
            scored_results = []
            for result in raw_results:
                # Calculate relevance score
                similarity_score = result.get("score", 0.0)
                
                # Apply modality filter
                metadata = result.get("metadata", {})
                chunk_modality = metadata.get("modality", "")
                
                if query.modality and query.modality != "multimodal":
                    if chunk_modality != query.modality:
                        continue
                
                # Apply page filter
                if query.page_number:
                    chunk_page = metadata.get("page_number", 0)
                    if chunk_page != query.page_number:
                        continue
                
                # Apply confidence threshold
                chunk_confidence = metadata.get("confidence", 0.0)
                if chunk_confidence < 0.3:  # Minimum confidence
                    continue
                
                # Calculate relevance score (weighted combination)
                relevance_score = self._calculate_relevance_score(
                    similarity_score,
                    chunk_confidence,
                    metadata
                )
                
                # Apply similarity threshold
                if relevance_score < query.similarity_threshold:
                    continue
                
                # Create RAGResult
                rag_result = RAGResult(
                    chunk=RAGChunk(**{
                        "id": str(uuid.uuid4()),  # Generate new ID
                        "document_id": metadata.get("document_id", ""),
                        "page_number": metadata.get("page_number", 1),
                        "text": result.get("text", ""),
                        "visual_description": metadata.get("visual_description", ""),
                        "bbox": metadata.get("bbox"),
                        "region_type": metadata.get("region_type"),
                        "modality": metadata.get("modality", "text"),
                        "agent_source": metadata.get("agent_source", "unknown"),
                        "confidence": chunk_confidence,
                        "metadata": metadata
                    }),
                    similarity_score=similarity_score,
                    relevance_score=relevance_score,
                    evidence_type=self._determine_evidence_type(metadata)
                )
                scored_results.append(rag_result)
            
            # Sort by relevance score
            scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Return top-k
            final_results = scored_results[:query.top_k]
            logger.info(f"Retrieved {len(final_results)} results for query: {query.query[:50]}...")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []
    
    def _calculate_relevance_score(self, similarity: float, confidence: float, 
                                  metadata: Dict[str, Any]) -> float:
        """Calculate weighted relevance score."""
        # Weight factors
        weights = {
            "similarity": 0.5,
            "confidence": 0.3,
            "source_quality": 0.2
        }
        
        # Source quality score
        source = metadata.get("agent_source", "")
        source_quality = {
            "fusion_agent": 1.0,
            "text_agent": 0.9,
            "vision_agent": 0.8,
            "ocr_engine": 0.7
        }.get(source, 0.5)
        
        # Calculate weighted score
        score = (
            similarity * weights["similarity"] +
            confidence * weights["confidence"] +
            source_quality * weights["source_quality"]
        )
        
        return min(max(score, 0.0), 1.0)
    
    def _determine_evidence_type(self, metadata: Dict[str, Any]) -> str:
        """Determine type of evidence."""
        modality = metadata.get("modality", "")
        region_type = metadata.get("region_type", "")
        
        if modality == "multimodal":
            return "text_and_visual"
        elif modality == "vision":
            if region_type == "table":
                return "tabular_visual"
            elif region_type == "figure":
                return "visual_diagram"
            else:
                return "visual"
        else:
            if region_type == "table":
                return "tabular_text"
            else:
                return "textual"
    
    async def generate_answer(self, query: str, results: List[RAGResult]) -> Tuple[str, float]:
        """Generate answer using LLM with retrieved evidence."""
        if not results:
            return "No relevant information found.", 0.0
        
        try:
            # Prepare context from results
            context_parts = []
            for i, result in enumerate(results[:5]):  # Use top 5 results
                chunk = result.chunk
                evidence_text = chunk.text or chunk.visual_description or ""
                
                if evidence_text:
                    context_parts.append(
                        f"[Evidence {i+1} from page {chunk.page_number}, "
                        f"confidence: {result.relevance_score:.2f}]: {evidence_text}"
                    )
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following evidence from the document, answer the question.

Question: {query}

Evidence:
{context}

Instructions:
1. Answer based ONLY on the provided evidence
2. If evidence is insufficient or contradictory, say so
3. Include page numbers when referencing evidence
4. State confidence level based on evidence quality

Answer:"""
            
            # Generate answer
            response = await llm_client.generate(
                prompt=prompt,
                system_prompt="You are a precise document analyst. Answer questions based ONLY on provided evidence.",
                temperature=0.1
            )
            
            # Calculate answer confidence
            confidence = self._calculate_answer_confidence(results)
            
            return response.strip(), confidence
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            return "Error generating answer.", 0.0
    
    def _calculate_answer_confidence(self, results: List[RAGResult]) -> float:
        """Calculate confidence for generated answer."""
        if not results:
            return 0.0
        
        # Use weighted average of top 3 results
        top_results = results[:3]
        weights = [0.5, 0.3, 0.2]
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(top_results):
            if i < len(weights):
                weight = weights[i]
                total_weighted_score += result.relevance_score * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        return 0.0
    
    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """Complete RAG query pipeline."""
        # Retrieve evidence
        results = self.retrieve(rag_query)
        
        # Generate answer
        answer, confidence = await self.generate_answer(rag_query.query, results)
        
        # Create traceability info
        traceability = {
            "total_results": len(results),
            "pages_referenced": list(set(r.chunk.page_number for r in results)),
            "modalities_used": list(set(r.chunk.modality for r in results)),
            "agent_sources": list(set(r.chunk.agent_source for r in results)),
            "average_confidence": np.mean([r.relevance_score for r in results]) if results else 0.0
        }
        
        return RAGResponse(
            query=rag_query.query,
            results=results,
            answer=answer,
            confidence=confidence,
            traceability=traceability
        )


# Global instance
retriever = MultiModalRetriever()
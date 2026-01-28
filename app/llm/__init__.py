"""
LLM and Embedding utilities for Phase 4.
"""
from app.llm.client import llm_client
from app.llm.embeddings import embedding_generator

__all__ = [
    'llm_client',
    'embedding_generator'
]
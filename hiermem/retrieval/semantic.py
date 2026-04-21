"""
Semantic Retrieval — Vector similarity search via ChromaDB.
"""

from typing import List, Dict, Optional
from hiermem.memory.vector_store import VectorStore


def semantic_search(vector_store: VectorStore, query: str,
                    top_k: int = 5, level_filter: Optional[str] = None) -> List[Dict]:
    """
    Search the vector store for semantically similar content.
    
    Args:
        vector_store: The ChromaDB-backed store
        query: Natural language query
        top_k: Number of results
        level_filter: Optional "L2" or "L3" to restrict search level
    """
    filters = {}
    if level_filter:
        filters["level"] = level_filter
    
    return vector_store.query(query_text=query, n_results=top_k, filters=filters or None)

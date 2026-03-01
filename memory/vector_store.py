"""
Vector Store — ChromaDB wrapper for L2/L3 storage and semantic search.
"""

import uuid
from typing import List, Optional, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

import config


class VectorStore:
    """
    ChromaDB-backed vector store for L2 summaries and L3 raw content.
    
    Supports:
      - Adding text with metadata
      - Semantic similarity search
      - Metadata-filtered queries
      - Retrieval by turn number
    """

    def __init__(self, collection_name: str = "hiermem",
                 persist_dir: str = None):
        self.collection_name = collection_name
        persist_dir = persist_dir or str(config.CHROMA_DB_PATH)
        
        if not HAS_CHROMADB:
            # Fallback: in-memory list-based store for testing
            self._fallback_store: List[Dict[str, Any]] = []
            self._use_fallback = True
            return
        
        self._use_fallback = False
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, text: str, metadata: Optional[Dict] = None):
        """Add a text chunk with metadata to the store."""
        doc_id = str(uuid.uuid4())[:12]
        metadata = metadata or {}
        
        if self._use_fallback:
            self._fallback_store.append({
                "id": doc_id, "text": text, "metadata": metadata
            })
            return doc_id
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id

    def query(self, query_text: str, n_results: int = 5,
              filters: Optional[Dict] = None) -> List[Dict]:
        """Semantic similarity search."""
        if self._use_fallback:
            return self._fallback_query(query_text, n_results, filters)
        
        kwargs = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if filters:
            kwargs["where"] = filters
        
        try:
            results = self.collection.query(**kwargs)
            
            items = []
            for i in range(len(results["documents"][0])):
                items.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
            return items
        except Exception:
            return []

    def get_by_metadata(self, filters: Dict, sort_by: str = "turn",
                        descending: bool = True, limit: int = 5) -> List[Dict]:
        """Get documents matching metadata filters."""
        if self._use_fallback:
            return self._fallback_get_by_metadata(filters, sort_by, descending, limit)
        
        try:
            results = self.collection.get(
                where=filters,
                limit=limit * 3  # over-fetch for sorting
            )
            
            items = []
            for i in range(len(results["documents"])):
                items.append({
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
            
            # Sort
            if sort_by:
                items.sort(
                    key=lambda x: x.get("metadata", {}).get(sort_by, 0),
                    reverse=descending
                )
            
            return items[:limit]
        except Exception:
            return []

    def count(self) -> int:
        if self._use_fallback:
            return len(self._fallback_store)
        return self.collection.count()

    def clear(self):
        """Clear all data."""
        if self._use_fallback:
            self._fallback_store.clear()
        else:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    # --- Fallback methods (no ChromaDB) ---

    def _fallback_query(self, query_text, n_results, filters):
        """Simple keyword matching fallback when ChromaDB isn't available."""
        query_words = set(query_text.lower().split())
        scored = []
        for doc in self._fallback_store:
            if filters:
                match = all(doc["metadata"].get(k) == v for k, v in filters.items())
                if not match:
                    continue
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": s[1]["text"], "metadata": s[1]["metadata"], "distance": 0}
                for s in scored[:n_results]]

    def _fallback_get_by_metadata(self, filters, sort_by, descending, limit):
        items = []
        for doc in self._fallback_store:
            match = all(doc["metadata"].get(k) == v for k, v in filters.items())
            if match:
                items.append({"text": doc["text"], "metadata": doc["metadata"]})
        items.sort(key=lambda x: x.get("metadata", {}).get(sort_by, 0), reverse=descending)
        return items[:limit]

"""
Embedding Model Wrapper — Sentence-transformers for local embeddings.

Used by ChromaDB (which has built-in embedding) and optionally for
manual similarity computations.
"""

from typing import List, Optional

from hiermem import config

# Lazy-loaded to avoid import cost when not needed
_model = None


def get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        except ImportError:
            _model = None
    return _model


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Embed a list of texts. Returns None if model unavailable."""
    model = get_embedding_model()
    if model is None:
        return None
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_single(text: str) -> Optional[List[float]]:
    """Embed a single text."""
    result = embed_texts([text])
    if result:
        return result[0]
    return None

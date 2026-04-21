"""
Keyword Retrieval — Exact and regex-based search.

Used for finding specific named entities, rules, and exact terms.
"""

import re
from typing import List, Dict


def keyword_search(query: str, documents: List[Dict], text_key: str = "text",
                   top_k: int = 5) -> List[Dict]:
    """
    Simple keyword matching search.
    
    Args:
        query: Search query
        documents: List of dicts with a text field
        text_key: Key in dict containing the text
        top_k: Max results to return
    """
    query_terms = [t.strip().lower() for t in query.split() if len(t.strip()) > 2]
    scored = []
    
    for doc in documents:
        text = doc.get(text_key, "").lower()
        score = sum(1 for term in query_terms if term in text)
        if score > 0:
            scored.append((score, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k]]


def regex_search(pattern: str, documents: List[Dict], text_key: str = "text") -> List[Dict]:
    """Search documents using a regex pattern."""
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return []
    
    return [doc for doc in documents if compiled.search(doc.get(text_key, ""))]

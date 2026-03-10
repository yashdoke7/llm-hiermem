"""
Retrieval Router — Adaptive strategy selection based on curator decisions.

Routes retrieval to the appropriate method(s) based on what the
curator agent decided.
"""

from typing import List, Tuple, Optional

from core.curator import CuratorDecision
from core.archive import HierarchicalArchive
from core.assembler import ContextChunk
from memory.vector_store import VectorStore
import config


class RetrievalRouter:
    """
    Routes retrieval requests to the appropriate strategy.
    
    Strategies:
      KEYWORD:   Search constraint store + L1 text by keyword
      HIERARCHY: Traverse L0 → L1 → L2 for selected segments
      SEMANTIC:  Vector similarity search in ChromaDB (L2/L3)
      HYBRID:    Combine hierarchy + semantic, deduplicate
      NONE:      No retrieval (early turns)
    """

    def __init__(self, archive: HierarchicalArchive, vector_store: VectorStore):
        self.archive = archive
        self.vector_store = vector_store

    def retrieve(self, decision: CuratorDecision
                 ) -> Tuple[List[ContextChunk], List[str]]:
        """
        Execute retrieval based on curator's decision.
        
        Returns:
            Tuple of (relevant_chunks, peripheral_summaries)
        """
        if decision.retrieval_strategy == "NONE":
            return [], []

        relevant_chunks = []
        peripheral_summaries = []

        # Fetch L1 segments selected by curator
        if decision.segments_to_fetch:
            l1_results = self.archive.get_l1_segments_batch(decision.segments_to_fetch)
            for seg_id, text in l1_results.items():
                relevant_chunks.append(
                    ContextChunk.from_text(text=text, source=f"L1:{seg_id}")
                )

        # Run semantic queries against ChromaDB
        if decision.semantic_queries:
            for query in decision.semantic_queries:
                results = self.archive.semantic_search(
                    query=query, top_k=config.SEMANTIC_TOP_K
                )
                for r in results:
                    # Avoid duplicates (check if similar text already included)
                    if not self._is_duplicate(r.get("text", ""), relevant_chunks):
                        relevant_chunks.append(
                            ContextChunk.from_text(
                                text=r["text"],
                                source=f"L{r.get('metadata', {}).get('level', '?')}:semantic"
                            )
                        )

        # Get peripheral summaries (1-line only from L0)
        for seg_id in decision.peripheral_segments:
            entry = self.archive.l0_directory.get(seg_id)
            if entry:
                peripheral_summaries.append(entry.label)

        # Fetch full L3 raw content for specifically requested turns
        for turn_num in getattr(decision, 'fetch_full_turns', []):
            content = self.archive.get_full_turn_content(turn_num)
            if content and not self._is_duplicate(content, relevant_chunks):
                # Insert at front so full content is prioritised in Zone 2
                relevant_chunks.insert(0, ContextChunk.from_text(
                    text=content, source=f"L3:turn_{turn_num}"
                ))

        return relevant_chunks, peripheral_summaries

    def _is_duplicate(self, text: str, existing: List[ContextChunk],
                      threshold: float = 0.8) -> bool:
        """Simple deduplication check based on text overlap."""
        text_words = set(text.lower().split())
        for chunk in existing:
            chunk_words = set(chunk.text.lower().split())
            if not text_words or not chunk_words:
                continue
            overlap = len(text_words & chunk_words) / max(len(text_words), len(chunk_words))
            if overlap > threshold:
                return True
        return False

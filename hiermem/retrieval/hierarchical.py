"""
Hierarchical Retrieval — Tree traversal through L0 → L1 → L2 → L3.
"""

from typing import List, Dict, Optional
from hiermem.core.archive import HierarchicalArchive


def hierarchical_retrieve(archive: HierarchicalArchive,
                          segment_ids: List[str],
                          depth: str = "L1") -> List[Dict]:
    """
    Retrieve context by traversing the hierarchy.
    
    Args:
        archive: The hierarchical archive
        segment_ids: Which segments to retrieve
        depth: How deep to go — "L1" (summaries only) or "L2" (chunk detail)
    """
    results = []
    
    for seg_id in segment_ids:
        l1_text = archive.get_l1_segment(seg_id)
        if l1_text:
            results.append({
                "text": l1_text,
                "source": f"L1:{seg_id}",
                "level": "L1"
            })
        
        if depth in ("L2", "L3"):
            # Get L2 summaries for turns in this segment
            entry = archive.l0_directory.get(seg_id)
            if entry:
                l2_results = archive.semantic_search(
                    query=entry.label,
                    top_k=5,
                    level="L2"
                )
                results.extend([
                    {"text": r["text"], "source": f"L2:{seg_id}", "level": "L2"}
                    for r in l2_results
                ])
    
    return results

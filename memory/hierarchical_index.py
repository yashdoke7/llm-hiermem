"""
Hierarchical Index — In-memory L0/L1 index management.

Lightweight wrapper used by the archive. L0 and L1 are small enough
to live entirely in Python dicts — no DB needed.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IndexEntry:
    """A single entry in the hierarchical index."""
    key: str
    summary: str
    turn_range: tuple  # (start, end)
    child_count: int = 0


class HierarchicalIndex:
    """
    Manages the L0/L1 index structure as pure Python dicts.
    
    This is intentionally simple — the complexity lives in
    archive.py which uses this as a building block.
    """

    def __init__(self):
        self.entries: Dict[str, IndexEntry] = {}

    def add(self, key: str, summary: str, turn_range: tuple, child_count: int = 0):
        self.entries[key] = IndexEntry(
            key=key, summary=summary,
            turn_range=turn_range, child_count=child_count
        )

    def get(self, key: str) -> Optional[IndexEntry]:
        return self.entries.get(key)

    def get_all_sorted(self) -> List[IndexEntry]:
        return sorted(self.entries.values(), key=lambda e: e.turn_range[0])

    def remove(self, key: str):
        self.entries.pop(key, None)

    def to_text(self) -> str:
        """Render as text (for curator context)."""
        entries = self.get_all_sorted()
        return "\n".join(f"{e.key}: \"{e.summary}\"" for e in entries)

    def __len__(self) -> int:
        return len(self.entries)

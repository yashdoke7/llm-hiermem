"""
Hierarchical Archive — Multi-level paged memory inspired by OS virtual memory.

Levels:
  L0: Topic Directory — one-line segment labels (~20 tokens each, always in curator context)
  L1: Segment Summaries — bullet-point summaries (~100 tokens each, fetched selectively)
  L2: Chunk Summaries — per-turn condensed summaries (in ChromaDB, semantic search)
  L3: Raw Content — full original text (in ChromaDB, exact retrieval)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

from memory.vector_store import VectorStore


@dataclass
class L0Entry:
    """One-line label for a segment in the topic directory."""
    segment_id: str
    label: str             # ~20 tokens: "Turns 1-10: setup, rules, Flask"
    turn_start: int
    turn_end: int
    is_merged: bool = False  # True if this is a super-segment (merged from multiple)


@dataclass
class L1Entry:
    """A single bullet point within a segment summary."""
    text: str
    turn: int
    importance: str = "normal"  # normal | high | critical


@dataclass
class L1Segment:
    """Collection of bullet-point summaries for a segment."""
    segment_id: str
    entries: List[L1Entry] = field(default_factory=list)

    def to_text(self) -> str:
        return "\n".join(f"• {e.text}" for e in self.entries)


class HierarchicalArchive:
    """
    Multi-level paged memory manager.
    
    The curator only ever sees L0 (the index). L1/L2/L3 are fetched
    on demand based on the curator's decisions.
    
    Capacity scales exponentially with depth:
      2-level: D × S turns (D=20 segments, S=10 turns/seg = 200 turns)
      3-level: With merging, effectively M × D × S turns
    """

    def __init__(self, vector_store: VectorStore,
                 max_l0_entries: int = 20,
                 segment_size: int = 10,
                 persist_dir: Optional[Path] = None):
        self.vector_store = vector_store
        self.max_l0_entries = max_l0_entries
        self.segment_size = segment_size
        self.persist_dir = persist_dir
        
        # L0: Topic Directory
        self.l0_directory: Dict[str, L0Entry] = {}
        
        # L1: Segment Summaries
        self.l1_segments: Dict[str, L1Segment] = {}
        
        # State tracking
        self.current_segment_id: str = "seg_001"
        self.current_segment_turn_count: int = 0
        self.total_segments_created: int = 1
        self.total_turns: int = 0
        
        # Initialize first segment
        self._init_segment(self.current_segment_id, turn_start=1)
        
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def add_turn(self, turn_number: int, user_msg: str, assistant_msg: str,
                 summary: str, l1_bullet: str):
        """
        Called after every conversation turn.
        
        Args:
            turn_number: Sequential turn number
            user_msg: Original user message
            assistant_msg: LLM response
            summary: Condensed turn summary (~50-100 tokens) for L2
            l1_bullet: One-line bullet point for L1 (~20 tokens)
        """
        self.total_turns = turn_number
        
        # L3: Store raw content in vector store
        raw_content = f"Turn {turn_number} User: {user_msg}\nAssistant: {assistant_msg}"
        self.vector_store.add(
            text=raw_content,
            metadata={"level": "L3", "turn": turn_number,
                      "segment": self.current_segment_id}
        )
        
        # L2: Store summary in vector store
        self.vector_store.add(
            text=summary,
            metadata={"level": "L2", "turn": turn_number,
                      "segment": self.current_segment_id}
        )
        
        # L1: Add bullet to current segment
        entry = L1Entry(text=l1_bullet, turn=turn_number)
        self.l1_segments[self.current_segment_id].entries.append(entry)
        
        # Update L0 label for current segment
        self._update_l0_label(self.current_segment_id)
        
        self.current_segment_turn_count += 1
        
        # Check if segment is full → create new segment
        if self.current_segment_turn_count >= self.segment_size:
            self._close_current_segment()
        
        # Check if L0 is over budget → merge oldest segments
        if len(self.l0_directory) > self.max_l0_entries:
            self._merge_oldest_segments()
        
        self._persist()

    def get_l0_directory_text(self) -> str:
        """Get the full L0 topic directory as text (for curator context)."""
        entries = sorted(self.l0_directory.values(), key=lambda e: e.turn_start)
        lines = [f"{e.segment_id}: \"{e.label}\"" for e in entries]
        return "\n".join(lines)

    def get_l1_segment(self, segment_id: str) -> Optional[str]:
        """Fetch L1 detail for a specific segment."""
        seg = self.l1_segments.get(segment_id)
        if seg:
            return seg.to_text()
        return None

    def get_l1_segments_batch(self, segment_ids: List[str]) -> Dict[str, str]:
        """Fetch multiple L1 segments at once."""
        results = {}
        for sid in segment_ids:
            text = self.get_l1_segment(sid)
            if text:
                results[sid] = text
        return results

    def semantic_search(self, query: str, top_k: int = 5,
                        level: Optional[str] = None) -> List[dict]:
        """Search L2/L3 via vector similarity."""
        filters = {}
        if level:
            filters["level"] = level
        return self.vector_store.query(query, n_results=top_k, filters=filters)

    def get_recent_turns(self, n: int = 3) -> List[dict]:
        """Get the N most recent turns (L3 raw content)."""
        return self.vector_store.get_by_metadata(
            filters={"level": "L3"},
            sort_by="turn",
            descending=True,
            limit=n
        )

    def get_full_turn_content(self, turn_number: int) -> Optional[str]:
        """Fetch the full L3 raw content for a specific turn number."""
        results = self.vector_store.get_by_metadata(
            filters={"level": "L3", "turn": turn_number},
            sort_by="turn",
            descending=False,
            limit=1
        )
        return results[0]["text"] if results else None

    def get_segment_ids(self) -> List[str]:
        """Get all segment IDs in order."""
        entries = sorted(self.l0_directory.values(), key=lambda e: e.turn_start)
        return [e.segment_id for e in entries]

    # --- Internal methods ---

    def _init_segment(self, segment_id: str, turn_start: int):
        """Initialize a new empty segment."""
        self.l0_directory[segment_id] = L0Entry(
            segment_id=segment_id,
            label=f"Turns {turn_start}-...: (in progress)",
            turn_start=turn_start,
            turn_end=turn_start
        )
        self.l1_segments[segment_id] = L1Segment(segment_id=segment_id)

    def _close_current_segment(self):
        """Close current segment and start a new one."""
        # Update end turn
        self.l0_directory[self.current_segment_id].turn_end = self.total_turns
        
        # Create new segment
        self.total_segments_created += 1
        new_id = f"seg_{self.total_segments_created:03d}"
        self.current_segment_id = new_id
        self.current_segment_turn_count = 0
        self._init_segment(new_id, turn_start=self.total_turns + 1)

    def _update_l0_label(self, segment_id: str):
        """Regenerate L0 label from L1 entries (simple keyword extraction)."""
        seg = self.l1_segments.get(segment_id)
        if not seg or not seg.entries:
            return
        
        entry = self.l0_directory[segment_id]
        # Use first and last bullet points to summarize
        first = seg.entries[0].text[:50]
        topics = first
        if len(seg.entries) > 1:
            last = seg.entries[-1].text[:50]
            topics = f"{first}; {last}"
        
        entry.label = f"Turns {entry.turn_start}-{self.total_turns}: {topics}"
        entry.turn_end = self.total_turns

    def _merge_oldest_segments(self):
        """Merge the two oldest segments into one super-segment."""
        entries = sorted(self.l0_directory.values(), key=lambda e: e.turn_start)
        if len(entries) < 2:
            return
        
        old1, old2 = entries[0], entries[1]
        
        # Create merged L0 entry
        merged_id = f"merged_{old1.segment_id}_{old2.segment_id}"
        merged_label = f"Turns {old1.turn_start}-{old2.turn_end}: (merged) {old1.label[:30]}..."
        
        self.l0_directory[merged_id] = L0Entry(
            segment_id=merged_id,
            label=merged_label,
            turn_start=old1.turn_start,
            turn_end=old2.turn_end,
            is_merged=True
        )
        
        # Merge L1 entries (keep only high-importance ones to save space)
        merged_entries = []
        for seg_id in [old1.segment_id, old2.segment_id]:
            seg = self.l1_segments.get(seg_id, L1Segment(segment_id=seg_id))
            # Keep critical entries, sample others
            for e in seg.entries:
                if e.importance in ("high", "critical"):
                    merged_entries.append(e)
                elif len(merged_entries) < self.segment_size:
                    merged_entries.append(e)
        
        self.l1_segments[merged_id] = L1Segment(
            segment_id=merged_id, entries=merged_entries
        )
        
        # Remove old entries
        del self.l0_directory[old1.segment_id]
        del self.l0_directory[old2.segment_id]
        if old1.segment_id in self.l1_segments:
            del self.l1_segments[old1.segment_id]
        if old2.segment_id in self.l1_segments:
            del self.l1_segments[old2.segment_id]

    def _persist(self):
        """Save L0 and L1 to disk."""
        if not self.persist_dir:
            return
        l0_data = {k: asdict(v) for k, v in self.l0_directory.items()}
        l1_data = {k: {"segment_id": v.segment_id,
                        "entries": [asdict(e) for e in v.entries]}
                   for k, v in self.l1_segments.items()}
        state = {
            "l0": l0_data, "l1": l1_data,
            "current_segment_id": self.current_segment_id,
            "current_segment_turn_count": self.current_segment_turn_count,
            "total_segments_created": self.total_segments_created,
            "total_turns": self.total_turns
        }
        (self.persist_dir / "archive_state.json").write_text(json.dumps(state, indent=2))

    def _load(self):
        """Load L0 and L1 from disk."""
        state_path = self.persist_dir / "archive_state.json"
        if not state_path.exists():
            return
        state = json.loads(state_path.read_text())
        self.l0_directory = {k: L0Entry(**v) for k, v in state["l0"].items()}
        self.l1_segments = {}
        for k, v in state["l1"].items():
            entries = [L1Entry(**e) for e in v["entries"]]
            self.l1_segments[k] = L1Segment(segment_id=v["segment_id"], entries=entries)
        self.current_segment_id = state["current_segment_id"]
        self.current_segment_turn_count = state["current_segment_turn_count"]
        self.total_segments_created = state["total_segments_created"]
        self.total_turns = state["total_turns"]

    def __repr__(self) -> str:
        return (f"HierarchicalArchive(segments={len(self.l0_directory)}, "
                f"total_turns={self.total_turns})")

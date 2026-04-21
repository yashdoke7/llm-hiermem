"""
Constraint Store — Always-checked priority memory for critical rules.

Constraints are tagged key-value pairs that are ALWAYS included in the
assembled context for the Main LLM. They represent rules, preferences,
limits, and identity facts that must never be forgotten.
"""

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Constraint:
    """A single constraint/rule that must be respected."""
    id: str
    text: str
    category: str           # RULE | PREFERENCE | LIMIT | IDENTITY | FACT
    source_turn: int        # Turn number when this was stated
    priority: int           # 1 (low) to 5 (critical)
    active: bool = True
    keywords: List[str] = field(default_factory=list)

    def to_display(self) -> str:
        return f"[{self.category}|P{self.priority}] {self.text}"


class ConstraintStore:
    """
    Manages critical rules and constraints that must always be present
    in the Main LLM's context.
    
    Design: Small, bounded, always-loaded. Like a TLB in OS terms.
    """

    def __init__(self, max_constraints: int = 20, persist_path: Optional[Path] = None):
        self.constraints: dict[str, Constraint] = {}
        self.max_constraints = max_constraints
        self.persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load(persist_path)

    def add(self, text: str, category: str, priority: int,
            source_turn: int = 0, keywords: Optional[List[str]] = None) -> str:
        """Add a new constraint. Returns constraint ID."""
        cid = str(uuid.uuid4())[:8]
        if keywords is None:
            keywords = self._extract_keywords(text)
        
        constraint = Constraint(
            id=cid, text=text, category=category.upper(),
            source_turn=source_turn, priority=priority, keywords=keywords
        )
        
        # If at capacity, remove lowest priority inactive or oldest low-priority
        if len(self.get_all_active()) >= self.max_constraints:
            self._evict_lowest_priority()
        
        self.constraints[cid] = constraint
        self._persist()
        return cid

    def deactivate(self, constraint_id: str) -> bool:
        """Deactivate a constraint (user says it no longer applies)."""
        if constraint_id in self.constraints:
            self.constraints[constraint_id].active = False
            self._persist()
            return True
        return False

    def get_all_active(self) -> List[Constraint]:
        """Get all active constraints, sorted by priority (highest first)."""
        active = [c for c in self.constraints.values() if c.active]
        return sorted(active, key=lambda c: c.priority, reverse=True)

    def check_violation(self, text: str) -> List[Constraint]:
        """Check if text potentially violates any active constraint."""
        violations = []
        text_lower = text.lower()
        for c in self.get_all_active():
            for kw in c.keywords:
                if kw.lower() in text_lower:
                    violations.append(c)
                    break
        return violations

    def get_display_text(self) -> str:
        """Get formatted text for inclusion in assembled context.

        Uses strong imperative framing so the main LLM treats constraints
        as overriding any conflicting patterns in retrieved context.
        """
        active = self.get_all_active()
        if not active:
            return "No active constraints."
        header = (
            "⚠️ MANDATORY RULES — These override ALL prior code patterns and examples.\n"
            "If earlier context shows code that conflicts with these rules, IGNORE that old code.\n"
        )
        lines = [f"{i+1}. {c.to_display()}" for i, c in enumerate(active)]
        return header + "\n".join(lines)

    def total_tokens_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        return len(self.get_display_text()) // 4

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from constraint text for violation checking."""
        keywords = []
        # Quoted strings (highest quality)
        keywords.extend(re.findall(r'"([^"]+)"', text))
        keywords.extend(re.findall(r"'([^']+)'", text))
        # File-like patterns (word.ext)
        keywords.extend(re.findall(r'\b[\w-]+\.\w+\b', text))
        # If no keywords found, use significant words (>4 chars), filtering stopwords
        if not keywords:
            stopwords = {
                "always", "never", "please", "should", "would", "could",
                "about", "their", "there", "these", "those", "which",
                "because", "every", "where", "being", "after", "before",
                "while", "other", "might", "still", "since", "under",
                "above", "below", "between", "through", "during", "until",
                "again", "further", "first", "second", "third", "given",
                "using", "based", "ensure", "include", "following", "without",
                "really", "quite", "often", "sometimes", "important",
                "confuse", "confuses",
            }
            words = text.split()
            keywords = [
                w.strip(".,!?;:")
                for w in words
                if len(w) > 4 and w.lower().strip(".,!?;:") not in stopwords
            ]
        return keywords[:10]

    def _evict_lowest_priority(self):
        """Remove the lowest priority constraint to make room."""
        active = self.get_all_active()
        if active:
            lowest = min(active, key=lambda c: (c.priority, -c.source_turn))
            del self.constraints[lowest.id]

    def _persist(self):
        """Save to disk if persist_path is set."""
        if self.persist_path:
            data = {cid: asdict(c) for cid, c in self.constraints.items()}
            self.persist_path.write_text(json.dumps(data, indent=2))

    def _load(self, path: Path):
        """Load from disk."""
        data = json.loads(path.read_text())
        for cid, cdata in data.items():
            self.constraints[cid] = Constraint(**cdata)

    def __len__(self) -> int:
        return len(self.get_all_active())

    def __repr__(self) -> str:
        return f"ConstraintStore(active={len(self)}, total={len(self.constraints)})"

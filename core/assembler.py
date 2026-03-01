"""
Context Assembler — Attention-aware context window construction.

Builds the final prompt for the Main LLM with 4 zones:
  Zone 1 (TOP):    Constraints — highest attention region
  Zone 2 (UPPER):  Relevant retrieved context
  Zone 3 (MIDDLE): Peripheral awareness — lowest attention region
  Zone 4 (BOTTOM): Current user prompt — high attention region

Based on "Lost in the Middle" (Liu et al., 2023): LLMs attend most
to the beginning and end of their context window.
"""

from typing import List, Optional
from dataclasses import dataclass

import config


@dataclass
class ContextChunk:
    """A piece of retrieved context to include."""
    text: str
    source: str     # "L1:seg_001" | "L2:turn_15" | "L3:turn_15" | "recent"
    tokens_est: int  # Estimated token count

    @classmethod
    def from_text(cls, text: str, source: str) -> "ContextChunk":
        return cls(text=text, source=source, tokens_est=len(text) // 4)


@dataclass
class AssembledContext:
    """The final assembled context ready for the Main LLM."""
    full_text: str
    total_tokens_est: int
    zone_breakdown: dict  # zone_name → token count
    sources_used: List[str]


class ContextAssembler:
    """
    Assembles retrieved context into an attention-optimized prompt.
    
    Key principle: Important things go at the TOP and BOTTOM of the
    context. Less important things go in the MIDDLE.
    """

    def __init__(self, total_budget: int = config.TOTAL_CONTEXT_BUDGET):
        self.total_budget = total_budget

    def assemble(self, constraints_text: str, relevant_chunks: List[ContextChunk],
                 peripheral_summaries: List[str], current_prompt: str,
                 recent_turns: Optional[List[ContextChunk]] = None,
                 system_prompt: str = "") -> AssembledContext:
        """
        Build the final context for the Main LLM.
        
        Args:
            constraints_text: Formatted constraints (always included)
            relevant_chunks: Retrieved relevant context pieces
            peripheral_summaries: 1-line summaries of other segments
            current_prompt: The user's current message
            recent_turns: Last N turns (always included)
            system_prompt: Optional system prompt prefix
        """
        sections = []
        token_counts = {}

        # Zone 1: Constraints (TOP — highest attention)
        zone1 = self._build_zone_1(constraints_text)
        sections.append(zone1)
        token_counts["constraints"] = len(zone1) // 4

        # Recent turns (after constraints, before retrieved context)
        if recent_turns:
            recent_text = self._build_recent_turns(recent_turns)
            sections.append(recent_text)
            token_counts["recent_turns"] = len(recent_text) // 4

        # Zone 2: Retrieved relevant context (UPPER-MIDDLE)
        zone2 = self._build_zone_2(relevant_chunks)
        if zone2:
            sections.append(zone2)
            token_counts["relevant_context"] = len(zone2) // 4

        # Zone 3: Peripheral awareness (LOWER-MIDDLE — lowest attention)
        if peripheral_summaries:
            zone3 = self._build_zone_3(peripheral_summaries)
            sections.append(zone3)
            token_counts["peripheral"] = len(zone3) // 4

        # Zone 4: Current prompt (BOTTOM — high attention)
        zone4 = self._build_zone_4(current_prompt)
        sections.append(zone4)
        token_counts["current_prompt"] = len(zone4) // 4

        full_text = "\n\n".join(sections)
        total_tokens = len(full_text) // 4

        # Truncate if over budget
        if total_tokens > self.total_budget:
            full_text = self._truncate_to_budget(
                constraints_text, relevant_chunks, peripheral_summaries,
                current_prompt, recent_turns
            )
            total_tokens = len(full_text) // 4

        sources = [c.source for c in relevant_chunks]
        if recent_turns:
            sources.extend([c.source for c in recent_turns])

        return AssembledContext(
            full_text=full_text,
            total_tokens_est=total_tokens,
            zone_breakdown=token_counts,
            sources_used=sources
        )

    def _build_zone_1(self, constraints_text: str) -> str:
        if not constraints_text or constraints_text == "No active constraints.":
            return ""
        return f"=== ACTIVE RULES & CONSTRAINTS ===\n{constraints_text}"

    def _build_recent_turns(self, recent_turns: List[ContextChunk]) -> str:
        texts = [chunk.text for chunk in reversed(recent_turns)]  # oldest first
        return "=== RECENT CONVERSATION ===\n" + "\n---\n".join(texts)

    def _build_zone_2(self, chunks: List[ContextChunk]) -> str:
        if not chunks:
            return ""
        texts = [chunk.text for chunk in chunks]
        return "=== RELEVANT CONTEXT ===\n" + "\n---\n".join(texts)

    def _build_zone_3(self, summaries: List[str]) -> str:
        lines = [f"• {s}" for s in summaries]
        return "=== OTHER CONTEXT (summaries only) ===\n" + "\n".join(lines)

    def _build_zone_4(self, current_prompt: str) -> str:
        return f"=== CURRENT REQUEST ===\n{current_prompt}"

    def _truncate_to_budget(self, constraints_text, relevant_chunks,
                             peripheral_summaries, current_prompt, recent_turns):
        """
        When over budget, remove in priority order:
        1. First drop peripheral summaries
        2. Then trim relevant chunks (remove lowest-relevance)
        3. Never drop constraints or current prompt
        """
        # Start with must-haves
        must_have = []
        if constraints_text and constraints_text != "No active constraints.":
            must_have.append(self._build_zone_1(constraints_text))
        if recent_turns:
            must_have.append(self._build_recent_turns(recent_turns))
        must_have.append(self._build_zone_4(current_prompt))

        must_have_tokens = sum(len(s) // 4 for s in must_have)
        remaining_budget = self.total_budget - must_have_tokens

        # Add as many relevant chunks as fit
        included_chunks = []
        for chunk in relevant_chunks:
            if remaining_budget - chunk.tokens_est > 0:
                included_chunks.append(chunk)
                remaining_budget -= chunk.tokens_est
            else:
                break

        # Assemble without peripheral
        sections = [s for s in must_have[:2]]  # constraints + recent
        if included_chunks:
            sections.append(self._build_zone_2(included_chunks))
        sections.append(must_have[-1])  # current prompt

        return "\n\n".join(sections)

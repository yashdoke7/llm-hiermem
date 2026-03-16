"""
Context Curator Agent — Stateless agent that selects relevant context.

The curator is the brain of the retrieval system. It sees:
  1. The current user prompt
  2. The L0 topic directory (tiny index of all segments)
  3. The constraint store

It decides:
  - Which retrieval strategy to use (keyword/hierarchy/semantic/hybrid)
  - Which segments to fetch from L1
  - What semantic queries to run against ChromaDB
  - Which segments to show as peripheral (1-line only)

CRITICAL: The curator is STATELESS. It has no memory of its own.
This prevents the nesting problem (curator itself degrading).
"""

import json
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from llm.client import LLMClient
import config


CURATOR_SYSTEM_PROMPT = (Path(config.PROMPTS_PATH) / "curator_system.txt").read_text() \
    if (Path(config.PROMPTS_PATH) / "curator_system.txt").exists() \
    else """You are a Context Curator. Your job is to select the most relevant context 
for the Main LLM to answer the user's current query.

You will receive:
1. The user's current prompt
2. A Topic Directory (compressed index of all past conversation segments)
3. The active Constraint Store

Your output must be ONLY a valid JSON object (no markdown, no explanation):
{
    "retrieval_strategy": "KEYWORD|HIERARCHY|SEMANTIC|HYBRID",
    "segments_to_fetch": ["seg_001", "seg_003"],
    "semantic_queries": ["query1", "query2"],
    "peripheral_segments": ["seg_002"],
    "fetch_full_turns": [],
    "reasoning": "brief explanation of choices"
}

Rules:
- Select AT MOST 4 segments for full fetch (token budget constraint)
- ALL constraints are auto-included — you don't need to select those
- If uncertain whether something is relevant, include it as peripheral
- Recent 2-3 turns are always auto-included — don't waste slots on them
- For the first few turns of a conversation, segments_to_fetch may be empty
- semantic_queries: write 1-3 short search queries to find relevant past content
- peripheral_segments: segments that might be tangentially relevant (1-line summary only)
- fetch_full_turns: list turn numbers when you need the FULL raw content of a specific turn
  (e.g. the actual code a user pasted, a file they shared). Use this for debugging scenarios
  where a summary is not enough — e.g. [7, 12] to get the raw content of turns 7 and 12.
- CONSTRAINT AWARENESS: If a constraint says "use X instead of Y" or "never use Y",
  AVOID fetching segments that primarily demonstrate Y (the old pattern).
  Prefer segments created AFTER constraints were set."""


@dataclass
class CuratorDecision:
    """Output of the curator's decision."""
    retrieval_strategy: str    # KEYWORD | HIERARCHY | SEMANTIC | HYBRID
    segments_to_fetch: List[str]
    semantic_queries: List[str]
    peripheral_segments: List[str]
    fetch_full_turns: List[int]  # Request L3 raw content for specific turn numbers
    reasoning: str

    @classmethod
    def empty(cls) -> "CuratorDecision":
        """Return an empty decision (for early turns with no history)."""
        return cls(
            retrieval_strategy="NONE",
            segments_to_fetch=[],
            semantic_queries=[],
            peripheral_segments=[],
            fetch_full_turns=[],
            reasoning="No history to retrieve from."
        )


class ContextCurator:
    """
    Stateless context selection agent.
    
    Operates on the L0 index + constraints only. Never sees full content.
    This is the key design that prevents curator degradation.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def select_context(self, user_prompt: str, l0_directory_text: str,
                       constraints_text: str, total_turns: int) -> CuratorDecision:
        """
        Decide what context to retrieve for the main LLM.
        
        Args:
            user_prompt: Current user message
            l0_directory_text: The full L0 topic directory as text
            constraints_text: Formatted constraint store content
            total_turns: How many turns have occurred so far
        
        Returns:
            CuratorDecision with retrieval instructions
        """
        # For very early conversations, skip curation
        if total_turns <= config.RECENT_TURNS_ALWAYS_INCLUDED:
            return CuratorDecision.empty()

        prompt = self._build_curator_prompt(
            user_prompt, l0_directory_text, constraints_text
        )

        response = self.llm.call(
            system_prompt=CURATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=config.CURATOR_MODEL,
            temperature=config.TEMPERATURE_CURATOR,
            max_tokens=500
        )

        return self._parse_decision(response)

    def _build_curator_prompt(self, user_prompt: str, l0_text: str,
                               constraints_text: str) -> str:
        """Assemble the curator's input prompt."""
        return f"""=== CURRENT USER PROMPT ===
{user_prompt}

=== TOPIC DIRECTORY (all conversation segments) ===
{l0_text if l0_text.strip() else "(No segments yet)"}

=== ACTIVE CONSTRAINTS ===
{constraints_text if constraints_text.strip() else "(No constraints set)"}

Select which segments and search queries are needed to answer this prompt.
Return ONLY valid JSON."""

    def _parse_decision(self, response: str) -> CuratorDecision:
        """Parse curator's JSON response into a CuratorDecision."""
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            
            data = json.loads(cleaned)
            return CuratorDecision(
                retrieval_strategy=data.get("retrieval_strategy", "SEMANTIC"),
                segments_to_fetch=data.get("segments_to_fetch", [])[:config.MAX_SEGMENTS_TO_FETCH],
                semantic_queries=data.get("semantic_queries", [])[:config.MAX_SEMANTIC_QUERIES],
                peripheral_segments=data.get("peripheral_segments", []),
                fetch_full_turns=[int(t) for t in data.get("fetch_full_turns", []) if str(t).isdigit()],
                reasoning=data.get("reasoning", "")
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: default to semantic search with user prompt as query
            return CuratorDecision(
                retrieval_strategy="SEMANTIC",
                segments_to_fetch=[],
                semantic_queries=[user_prompt[:200] if 'user_prompt' in dir() else ""],
                peripheral_segments=[],
                fetch_full_turns=[],
                reasoning="Failed to parse curator response, falling back to semantic search."
            )

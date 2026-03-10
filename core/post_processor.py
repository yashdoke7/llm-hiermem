"""
Post-Processor — Runs after every Main LLM response.

Responsibilities:
  1. Check for constraint violations (LLM-based + keyword fallback)
  2. Extract new constraints from user messages
  3. Generate turn summary for L1/L2
  4. Update the hierarchical archive
"""

import json
import logging
import re
from typing import List, Optional, Tuple

from llm.client import LLMClient
from core.constraint_store import Constraint, ConstraintStore
from core.archive import HierarchicalArchive
import config

logger = logging.getLogger(__name__)


SUMMARIZER_PROMPT = """Summarize this conversation turn in ONE concise sentence (max 30 words).
Focus on: decisions made, actions taken, facts stated, rules set.

Turn:
User: {user_msg}
Assistant: {assistant_msg}

One-sentence summary:"""

CONSTRAINT_EXTRACTOR_PROMPT = """Extract constraints from this user message. A constraint is ANY of these:
- A rule: "never do X", "always do Y", "don't do Z"
- A preference: "I prefer X", "use X not Y", "I like X better"
- A limit: "keep it under X", "max X"
- A fact to remember: "my name is X", "we use X", "remember that X"

User message: {user_msg}

Return a JSON list. Example:
[{{"text": "User prefers Python over JavaScript", "category": "PREFERENCE", "priority": 3}}]

If truly NOTHING to extract, return: []
Return ONLY the JSON list, nothing else."""


class PostProcessor:
    """
    Post-response pipeline that maintains memory integrity.
    
    Runs after every Main LLM response to:
    - Catch constraint violations before delivery
    - Keep memory stores updated
    - Extract new constraints automatically
    """

    def __init__(self, llm_client: LLMClient, constraint_store: ConstraintStore,
                 archive: HierarchicalArchive):
        self.llm = llm_client
        self.constraint_store = constraint_store
        self.archive = archive

    def process(self, turn_number: int, user_msg: str,
                assistant_msg: str,
                retry_llm=None, retry_context: str = None,
                system_prompt: str = "You are a helpful assistant.") -> Tuple[str, List[str]]:
        """
        Full post-processing pipeline.
        
        Args:
            turn_number: Current turn number
            user_msg: User's message this turn
            assistant_msg: Main LLM's response
            retry_llm: LLM client for violation retry (optional)
            retry_context: The context that was sent to produce assistant_msg (for retry)
            system_prompt: System prompt for retry calls
            
        Returns:
            Tuple of (possibly modified response, list of warnings)
        """
        warnings = []

        # 1. Check constraint violations (LLM-based) and retry if possible
        if config.ENABLE_VIOLATION_CHECK:
            violations = self._check_violations_llm(assistant_msg)
            if violations:
                warning_text = self._format_violation_warning(violations)
                # Retry once with violation feedback injected
                if retry_llm and retry_context and config.ENABLE_VIOLATION_RETRY:
                    retried, retry_warning = self._retry_with_violation_feedback(
                        retry_llm, retry_context, assistant_msg,
                        violations, system_prompt
                    )
                    if retried:
                        assistant_msg = retried
                        warnings.append(f"violation_retry: {len(violations)} violations corrected")
                    else:
                        warnings.append(warning_text)
                else:
                    warnings.append(warning_text)

        # 2. Extract new constraints from user message
        if config.ENABLE_CONSTRAINT_EXTRACTION:
            new_constraints = self._extract_constraints(user_msg, turn_number)
            for c in new_constraints:
                self.constraint_store.add(**c)

        # 3. Generate summaries and update archive
        if config.ENABLE_AUTO_SUMMARIZE:
            summary, bullet = self._generate_summaries(user_msg, assistant_msg)
            self.archive.add_turn(
                turn_number=turn_number,
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                summary=summary,
                l1_bullet=bullet
            )

        return assistant_msg, warnings

    def _extract_constraints(self, user_msg: str, turn_number: int) -> List[dict]:
        """Use LLM to extract constraints from user message."""
        # Let the LLM extractor handle all cases — it returns [] for non-constraints.
        # Regex pre-check was removed: it blocked casual constraints like
        # "btw use rupees here" or "we switched from pandas to polars" that
        # contain no explicit signal words but are still valid user preferences.

        prompt = CONSTRAINT_EXTRACTOR_PROMPT.format(user_msg=user_msg)
        
        try:
            response = self.llm.call(
                system_prompt="You extract constraints from text. Return only valid JSON.",
                user_prompt=prompt,
                model=config.SUMMARIZER_MODEL,
                temperature=0.0,
                max_tokens=300
            )
            
            cleaned = response.strip()
            # Strip markdown code fences
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            
            # Try to find JSON array in the response
            bracket_start = cleaned.find("[")
            bracket_end = cleaned.rfind("]")
            if bracket_start != -1 and bracket_end != -1:
                cleaned = cleaned[bracket_start:bracket_end + 1]
            
            constraints_data = json.loads(cleaned)
            if not isinstance(constraints_data, list):
                return []
            
            return [
                {
                    "text": c["text"],
                    "category": c.get("category", "RULE"),
                    "priority": c.get("priority", 3),
                    "source_turn": turn_number
                }
                for c in constraints_data
                if "text" in c
            ]
        except Exception as e:
            import logging
            logging.debug(f"Constraint extraction failed: {e}, response was: {response[:200] if 'response' in dir() else 'N/A'}")
            return []

    def _generate_summaries(self, user_msg: str, assistant_msg: str) -> Tuple[str, str]:
        """Generate L2 summary and L1 bullet point."""
        prompt = SUMMARIZER_PROMPT.format(
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500]
        )
        
        try:
            summary = self.llm.call(
                system_prompt="You are a concise summarizer.",
                user_prompt=prompt,
                model=config.SUMMARIZER_MODEL,
                temperature=0.0,
                max_tokens=100
            )
            # L1 bullet is even shorter — first 100 chars of summary
            bullet = summary.strip()[:100]
            return summary.strip(), bullet
        except Exception:
            # Fallback: simple truncation
            fallback = f"User asked about: {user_msg[:50]}... Assistant responded with: {assistant_msg[:50]}..."
            return fallback, fallback[:100]

    def _check_violations_llm(self, response: str) -> List[Constraint]:
        """Use LLM to accurately check if response violates active constraints."""
        active = self.constraint_store.get_all_active()
        if not active:
            return []

        constraints_numbered = "\n".join(
            f"{i+1}. {c.text}" for i, c in enumerate(active)
        )
        prompt = (
            f"Check if this assistant response violates any of these rules.\n\n"
            f"Rules:\n{constraints_numbered}\n\n"
            f"Response:\n{response[:800]}\n\n"
            f"For each rule, output its number ONLY if it is clearly VIOLATED.\n"
            f"If all rules are followed, output: NONE\n"
            f"Output only rule numbers (comma-separated) or NONE:"
        )

        try:
            result = self.llm.call(
                system_prompt="You are a strict constraint violation checker. Only flag clear violations.",
                user_prompt=prompt,
                model=config.SUMMARIZER_MODEL,
                temperature=0.0,
                max_tokens=50
            )
            result = result.strip().upper()
            if "NONE" in result or not result:
                return []

            violations = []
            for num_str in re.findall(r'\d+', result):
                idx = int(num_str) - 1
                if 0 <= idx < len(active):
                    violations.append(active[idx])
            return violations
        except Exception as e:
            logger.debug(f"LLM violation check failed: {e}")
            return []

    def _format_violation_warning(self, violations) -> str:
        """Format constraint violations into a warning message."""
        lines = ["⚠️ POTENTIAL CONSTRAINT VIOLATIONS DETECTED:"]
        for v in violations:
            lines.append(f"  - [{v.category}] {v.text}")
        return "\n".join(lines)

    def _retry_with_violation_feedback(self, llm, context: str,
                                        original_response: str,
                                        violations, system_prompt: str) -> Tuple[Optional[str], str]:
        """Retry LLM call with violation feedback injected."""
        violation_list = "\n".join(f"- {v.text}" for v in violations)
        retry_prompt = (
            f"{context}\n\n"
            f"=== IMPORTANT: YOUR PREVIOUS RESPONSE VIOLATED THESE RULES ===\n"
            f"{violation_list}\n\n"
            f"Please regenerate your response while strictly following ALL of the above rules."
        )
        try:
            retried = llm.call(
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                model=config.MAIN_LLM_MODEL,
                temperature=config.TEMPERATURE_MAIN,
            )
            # Verify retry actually fixed it
            still_violated = self._check_violations_llm(retried)
            if len(still_violated) < len(violations):
                return retried, "retry_improved"
            return retried, "retry_attempted"
        except Exception:
            return None, "retry_failed"

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

from hiermem.llm.client import LLMClient
from hiermem.core.constraint_store import Constraint, ConstraintStore
from hiermem.core.archive import HierarchicalArchive
from hiermem import config

logger = logging.getLogger(__name__)


SUMMARIZER_PROMPT = """Summarize this conversation turn in ONE concise sentence (max 30 words).
Focus on: decisions made, actions taken, facts stated, rules set.

Turn:
User: {user_msg}
Assistant: {assistant_msg}

One-sentence summary:"""

CONSTRAINT_EXTRACTOR_PROMPT = """Extract constraints from the user message below. A constraint is a rule, preference, or fact that must be followed in future responses.

If there are NO clear constraints, you MUST return: []

If there are constraints, return a JSON array of objects. Example format:
[{{"text": "Extracted rule here", "category": "RULE", "priority": 3}}]

--- USER MESSAGE START ---
{user_msg}
--- USER MESSAGE END ---

Return ONLY valid JSON."""


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
                system_prompt: str = "You are a helpful assistant.") -> Tuple[str, List[str], dict]:
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
            Tuple of (possibly modified response, list of warnings, telemetry dict)
            
        telemetry dict keys:
            summary_tokens   int  — tokens used by summarization call
            extract_tokens   int  — tokens used by constraint extraction call
            retry_fired      bool — whether a violation retry was triggered
            retry_tokens     int  — tokens used by retry call (0 if not fired)
        """
        warnings = []
        telemetry = {
            "summary_tokens":  0,
            "extract_tokens":  0,
            "retry_fired":     False,
            "retry_tokens":    0,
        }

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
                        # Measure retry tokens: context + retry prompt overhead + response
                        _retry_prompt_overhead = 200  # violation list + instruction text
                        telemetry["retry_tokens"] = (
                            len(retry_context) // 4
                            + _retry_prompt_overhead
                            + len(retried) // 4
                        )
                        telemetry["retry_fired"] = True
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
            # Token estimate for extraction call: input prompt + response
            telemetry["extract_tokens"] = (
                len(CONSTRAINT_EXTRACTOR_PROMPT.format(user_msg=user_msg)) // 4
                + 80  # typical JSON list response
            )

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
            # Token estimate for summarization call: input prompt + response
            telemetry["summary_tokens"] = (
                len(SUMMARIZER_PROMPT.format(
                    user_msg=user_msg[:500], assistant_msg=assistant_msg[:500]
                )) // 4
                + 40  # typical one-sentence summary
            )

        return assistant_msg, warnings, telemetry

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
                max_tokens=config.MAX_TOKENS_CURATOR,
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
                system_prompt="You summarize conversation turns concisely.",
                user_prompt=prompt,
                model=config.SUMMARIZER_MODEL,
                temperature=0.0,
                max_tokens=config.MAX_TOKENS_SUMMARIZER
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
                max_tokens=config.MAX_TOKENS_CURATOR
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
                max_tokens=config.MAX_TOKENS_MAIN,
            )
            # Verify retry actually fixed it
            still_violated = self._check_violations_llm(retried)
            if len(still_violated) < len(violations):
                return retried, "retry_improved"
            return retried, "retry_attempted"
        except Exception:
            return None, "retry_failed"
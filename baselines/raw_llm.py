"""
Baseline 1: Raw LLM — Full conversation history in context window.

No memory management. Just appends all turns to the context until
the window is full, then truncates from the beginning.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

from llm.client import LLMClient
from llm.token_counter import count_tokens
import config


def _strip_hallucinated_continuation(text: str) -> str:
    """Remove fake User:/Assistant: exchanges the LLM hallucinates."""
    match = re.search(r'\n\s*User\s*:', text)
    if match:
        return text[:match.start()].rstrip()
    return text


@dataclass
class RawLLMTurnResult:
    turn_number: int
    user_message: str
    assistant_response: str
    prompt_tokens: int
    response_tokens: int
    context_tokens: int  # for legacy support


class RawLLMBaseline:
    """
    Simplest baseline: dump everything into context.
    Truncate from the oldest when it doesn't fit.
    """

    def __init__(self, provider: str = None, model: str = None,
                 max_context_tokens: int = None):
        self.llm = LLMClient(provider=provider or config.DEFAULT_PROVIDER)
        self.model = model or config.MAIN_LLM_MODEL
        self.max_context_tokens = max_context_tokens or config.RAW_LLM_CONTEXT_BUDGET
        self.conversation_history: List[dict] = []
        self.turn_count = 0

    def process_turn(self, user_message: str,
                     system_prompt: str = "You are a helpful assistant.") -> RawLLMTurnResult:
        self.turn_count += 1
        self.conversation_history.append({"role": "user", "content": user_message})

        # Build context from history, truncate oldest if over budget
        context = self._build_context(system_prompt)

        response = self.llm.call(
            system_prompt=system_prompt,
            user_prompt=context,
            model=self.model,
            temperature=config.TEMPERATURE_MAIN,
        )

        # Strip hallucinated multi-turn continuations
        response = _strip_hallucinated_continuation(response)

        self.conversation_history.append({"role": "assistant", "content": response})

        return RawLLMTurnResult(
            turn_number=self.turn_count,
            user_message=user_message,
            assistant_response=response,
            prompt_tokens=count_tokens(context),
            response_tokens=count_tokens(response),
            context_tokens=count_tokens(context)
        )

    def _build_context(self, system_prompt: str) -> str:
        """Build context from history, truncating oldest turns if necessary."""
        lines = []
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")

        context = "\n\n".join(lines)

        # Truncate from beginning if over budget
        while count_tokens(context) > self.max_context_tokens and len(lines) > 1:
            lines.pop(0)
            context = "\n\n".join(lines)

        return context

    def reset(self):
        self.conversation_history.clear()
        self.turn_count = 0

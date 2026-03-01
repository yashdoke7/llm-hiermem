"""
Baseline 3: RAG + Rolling Summary.

Maintains a rolling summary of the conversation that gets
prepended to each prompt, plus standard RAG retrieval.
"""

from typing import List
from dataclasses import dataclass

from llm.client import LLMClient
from llm.token_counter import count_tokens
from memory.vector_store import VectorStore
import config


@dataclass
class RAGSummaryTurnResult:
    turn_number: int
    user_message: str
    assistant_response: str
    context_tokens: int
    summary_tokens: int


class RAGSummaryBaseline:
    """
    RAG + rolling conversation summary prepended to context.
    """

    def __init__(self, provider: str = None, model: str = None,
                 top_k: int = 5, max_context_tokens: int = 6000,
                 max_summary_tokens: int = 500):
        self.llm = LLMClient(provider=provider or config.DEFAULT_PROVIDER)
        self.model = model or config.MAIN_LLM_MODEL
        self.vector_store = VectorStore(collection_name="rag_summary_baseline")
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.max_summary_tokens = max_summary_tokens
        self.rolling_summary = ""
        self.turn_count = 0

    def process_turn(self, user_message: str,
                     system_prompt: str = "You are a helpful assistant.") -> RAGSummaryTurnResult:
        self.turn_count += 1

        # Retrieve relevant past turns
        retrieved = []
        if self.vector_store.count() > 0:
            retrieved = self.vector_store.query(
                query_text=user_message, n_results=self.top_k
            )

        # Build context with summary + retrieved + current
        context_parts = []
        if self.rolling_summary:
            context_parts.append(f"=== CONVERSATION SUMMARY ===\n{self.rolling_summary}")
        if retrieved:
            context_parts.append("=== RELEVANT PAST CONTEXT ===")
            for r in retrieved:
                context_parts.append(r.get("text", ""))
        context_parts.append(f"\n=== CURRENT REQUEST ===\n{user_message}")
        context = "\n---\n".join(context_parts)

        response = self.llm.call(
            system_prompt=system_prompt,
            user_prompt=context,
            model=self.model,
            temperature=config.TEMPERATURE_MAIN,
        )

        # Store turn
        self.vector_store.add(
            text=f"User: {user_message}\nAssistant: {response}",
            metadata={"turn": self.turn_count}
        )

        # Update rolling summary
        self._update_summary(user_message, response)

        return RAGSummaryTurnResult(
            turn_number=self.turn_count,
            user_message=user_message,
            assistant_response=response,
            context_tokens=count_tokens(context),
            summary_tokens=count_tokens(self.rolling_summary)
        )

    def _update_summary(self, user_msg: str, assistant_msg: str):
        """Update the rolling summary with this turn's information."""
        prompt = f"""Current summary: {self.rolling_summary or '(empty)'}

New turn:
User: {user_msg[:300]}
Assistant: {assistant_msg[:300]}

Update the summary to include key information from this turn.
Keep total summary under 200 words. Focus on: decisions, facts, rules, constraints.
Updated summary:"""

        self.rolling_summary = self.llm.call(
            system_prompt="You maintain a concise conversation summary.",
            user_prompt=prompt,
            model=config.SUMMARIZER_MODEL,
            temperature=0.0,
            max_tokens=300
        ).strip()

    def reset(self):
        self.vector_store.clear()
        self.rolling_summary = ""
        self.turn_count = 0

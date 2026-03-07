"""
Baseline 2: Standard RAG — Vector similarity retrieval.

Stores all turns in ChromaDB. For each new turn, retrieves
the top-k most similar past turns and includes them in context.
"""

from typing import List
from dataclasses import dataclass

from llm.client import LLMClient
from llm.token_counter import count_tokens
from memory.vector_store import VectorStore
import config


@dataclass
class RAGTurnResult:
    turn_number: int
    user_message: str
    assistant_response: str
    context_tokens: int
    chunks_retrieved: int


class RAGBaseline:
    """
    Standard RAG baseline: embed all turns, retrieve top-k similar on each query.
    """

    def __init__(self, provider: str = None, model: str = None,
                 top_k: int = 5, max_context_tokens: int = 6000):
        self.llm = LLMClient(provider=provider or config.DEFAULT_PROVIDER)
        self.model = model or config.MAIN_LLM_MODEL
        self.vector_store = VectorStore(collection_name="rag_baseline")
        self.vector_store.clear()  # fresh store per conversation
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.turn_count = 0

    def process_turn(self, user_message: str,
                     system_prompt: str = "You are a helpful assistant.") -> RAGTurnResult:
        self.turn_count += 1

        # Retrieve relevant past turns
        retrieved = []
        if self.vector_store.count() > 0:
            retrieved = self.vector_store.query(
                query_text=user_message, n_results=self.top_k
            )

        # Build context
        context_parts = []
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

        # Store this turn for future retrieval
        self.vector_store.add(
            text=f"User: {user_message}\nAssistant: {response}",
            metadata={"turn": self.turn_count}
        )

        return RAGTurnResult(
            turn_number=self.turn_count,
            user_message=user_message,
            assistant_response=response,
            context_tokens=count_tokens(context),
            chunks_retrieved=len(retrieved)
        )

    def reset(self):
        self.vector_store.clear()
        self.turn_count = 0

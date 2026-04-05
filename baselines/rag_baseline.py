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
    prompt_tokens: int
    response_tokens: int
    context_tokens: int
    chunks_retrieved: int
    pipeline_details: dict = None


class RAGBaseline:
    """
    Standard RAG baseline: embed all turns, retrieve top-k similar on each query.
    """

    def __init__(self, provider: str = None, model: str = None,
                 top_k: int = 20, max_context_tokens: int = None):
        self.llm = LLMClient(provider=provider or config.DEFAULT_PROVIDER)
        self.model = model or config.MAIN_LLM_MODEL
        self.vector_store = VectorStore(collection_name="rag_baseline")
        self.vector_store.clear()  # fresh store per conversation
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens or config.RAG_CONTEXT_BUDGET
        self.turn_count = 0

    def process_turn(self, user_message: str,
                     system_prompt: str = "You are a helpful assistant.") -> RAGTurnResult:
        self.turn_count += 1

        # Retrieve relevant past turns
        retrieved = []
        has_index = self.vector_store.count() > 0
        retrieval_query_count = 1 if has_index else 0
        if has_index:
            retrieved = self.vector_store.query(
                query_text=user_message, n_results=self.top_k
            )

        # Build context, respecting token budget
        current_req_tokens = count_tokens(f"\n=== CURRENT REQUEST ===\n{user_message}")
        budget = self.max_context_tokens - current_req_tokens - 50  # 50 for header/separators

        context_parts = []
        if retrieved:
            header = "=== RELEVANT PAST CONTEXT ==="
            budget -= count_tokens(header + "\n---\n")
            context_parts.append(header)
            for r in retrieved:
                chunk = r.get("text", "")
                chunk_tok = count_tokens(chunk + "\n---\n")
                if budget - chunk_tok >= 0:
                    context_parts.append(chunk)
                    budget -= chunk_tok
                else:
                    break  # budget exhausted, skip remaining chunks
        context_parts.append(f"\n=== CURRENT REQUEST ===\n{user_message}")
        context = "\n---\n".join(context_parts)

        response = self.llm.call(
            system_prompt=system_prompt,
            user_prompt=context,
            model=self.model,
            temperature=config.TEMPERATURE_MAIN,
            max_tokens=config.MAX_TOKENS_MAIN,
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
            prompt_tokens=count_tokens(context),
            response_tokens=count_tokens(response),
            context_tokens=count_tokens(context),
            chunks_retrieved=len(retrieved),
            pipeline_details={
                # Legacy field kept for backward compatibility with old analysis scripts.
                "vector_queries": len(retrieved),
                # New explicit retrieval telemetry for compute accounting.
                "retrieval_query_count": retrieval_query_count,
                "retrieved_chunks_count": len(retrieved),
                "embedding_requests": retrieval_query_count,
                "semantic_queries": [user_message] if retrieval_query_count else [],
                "sources_used": [f"semantic:turn_{r.get('metadata',{}).get('turn','?')}" for r in retrieved],
                "context_tokens_used": count_tokens(context),
                "response_tokens": count_tokens(response),
            }
        )

    def reset(self):
        self.vector_store.clear()
        self.turn_count = 0

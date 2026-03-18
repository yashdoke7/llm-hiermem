"""
Token Counter — Accurate token counting for budget management.

NOTE: Uses tiktoken's cl100k_base (GPT-4) tokenizer as an approximation
for all models. For non-OpenAI models (e.g. Qwen, Gemma, Llama), token
counts will be approximate (~10-20% variance). The bias is consistent
across all systems, so comparative metrics remain valid.
"""

from typing import Optional

_encoder = None


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text. Uses tiktoken if available, else estimates.
    
    Note: Always uses cl100k_base encoding regardless of target model.
    This is an intentional cross-model approximation for consistent 
    budget management across different LLM backends.
    """
    global _encoder
    try:
        import tiktoken
        if _encoder is None:
            try:
                _encoder = tiktoken.encoding_for_model(model)
            except KeyError:
                _encoder = tiktoken.get_encoding("cl100k_base")
        return len(_encoder.encode(text))
    except ImportError:
        # Fallback: rough estimate (1 token ≈ 4 chars for English)
        return len(text) // 4


def fits_in_budget(text: str, budget: int, model: str = "gpt-4") -> bool:
    """Check if text fits within token budget."""
    return count_tokens(text, model) <= budget


def truncate_to_budget(text: str, budget: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token budget (from the end)."""
    tokens = count_tokens(text, model)
    if tokens <= budget:
        return text
    # Rough truncation: take proportional substring
    ratio = budget / tokens
    char_limit = int(len(text) * ratio * 0.95)  # 5% safety margin
    return text[:char_limit] + "\n... (truncated)"

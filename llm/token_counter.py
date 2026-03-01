"""
Token Counter — Accurate token counting for budget management.
"""

from typing import Optional

_encoder = None


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text. Uses tiktoken if available, else estimates.
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

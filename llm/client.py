"""
LLM Client — Multi-provider wrapper for LLM API calls.

Supports: Groq, OpenAI, Anthropic, Google, Ollama (local).
Uses litellm for unified API when available, falls back to direct calls.
Includes sliding-window rate limiter (modeled on llm-reasoning-pipeline).
"""

import time
import re
import json
from typing import Optional, List, Tuple

import config


class TokenBudget:
    """Sliding-window rate limiter for API free tiers.
    
    Tracks actual token usage in a 60s window and waits proactively
    when approaching limits, rather than reacting to 429 errors.
    """

    def __init__(self, tpm_limit: int = 12000, rpm_limit: int = 30,
                 min_gap: float = 5.0):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.min_gap = min_gap
        self._requests: List[float] = []
        self._tokens: List[Tuple[float, int]] = []
        self._last_request: float = 0.0

    def _clean(self, window: float = 60.0):
        now = time.time()
        self._requests = [t for t in self._requests if now - t < window]
        self._tokens = [(t, n) for t, n in self._tokens if now - t < window]

    def wait_if_needed(self, estimated_tokens: int = 500):
        """Block until it's safe to make another request."""
        # Enforce minimum gap between requests
        now = time.time()
        since_last = now - self._last_request
        if since_last < self.min_gap:
            time.sleep(self.min_gap - since_last)

        self._clean()

        # Check RPM: trigger at 60% to be safe
        rpm_threshold = int(self.rpm_limit * 0.6)
        if len(self._requests) >= rpm_threshold:
            sleep = 62 - (time.time() - self._requests[0])
            if sleep > 0:
                print(f"  [Rate limit] RPM pause {sleep:.0f}s ({len(self._requests)}/{self.rpm_limit} RPM)")
                time.sleep(sleep)
            self._clean()

        # Check TPM: trigger at 60% to be safe
        used = sum(n for _, n in self._tokens)
        tpm_threshold = int(self.tpm_limit * 0.6)
        if used + estimated_tokens >= tpm_threshold:
            if self._tokens:
                sleep = 62 - (time.time() - self._tokens[0][0])
                if sleep > 0:
                    print(f"  [Rate limit] TPM pause {sleep:.0f}s ({used}/{self.tpm_limit} TPM)")
                    time.sleep(sleep)
            self._clean()

    def record(self, tokens: int):
        """Record a completed request's token usage."""
        now = time.time()
        self._last_request = now
        self._requests.append(now)
        self._tokens.append((now, tokens))


# Shared rate limiters per provider (so all LLMClient instances share the budget)
_budgets = {}


def _get_budget(provider: str, model: str = "") -> Optional[TokenBudget]:
    """Get or create a shared TokenBudget for the provider+model combination."""
    if provider == "ollama":
        return None  # No rate limiting for local
    
    budget_key = f"{provider}:{model}" if model else provider
    if budget_key not in _budgets:
        if provider == "groq":
            # Per-model limits (free tier) — https://console.groq.com/docs/rate-limits
            if "70b" in model or "versatile" in model:
                _budgets[budget_key] = TokenBudget(tpm_limit=6000, rpm_limit=30, min_gap=3.0)
            elif "qwq" in model or "32b" in model:
                _budgets[budget_key] = TokenBudget(tpm_limit=6000, rpm_limit=30, min_gap=3.0)
            else:  # 8b-instant and others
                _budgets[budget_key] = TokenBudget(tpm_limit=20000, rpm_limit=30, min_gap=2.0)
        elif provider == "openai":
            _budgets[budget_key] = TokenBudget(tpm_limit=90000, rpm_limit=60, min_gap=1.0)
        elif provider == "google":
            # Gemini free tier: 15 RPM, 250k TPM → stay under 10 RPM for safety
            _budgets[budget_key] = TokenBudget(tpm_limit=250000, rpm_limit=10, min_gap=7.0)
        else:
            _budgets[budget_key] = TokenBudget(tpm_limit=60000, rpm_limit=60, min_gap=1.0)
    return _budgets[budget_key]


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Usage:
        client = LLMClient(provider="groq")
        response = client.call(
            system_prompt="You are helpful.",
            user_prompt="Hello!",
            model="llama-3.1-8b-instant"
        )
    """

    def __init__(self, provider: str = None):
        self.provider = provider or config.DEFAULT_PROVIDER
        self._call_count = 0

    def call(self, system_prompt: str, user_prompt: str,
             model: str = None, temperature: float = 0.3,
             max_tokens: int = 4096) -> str:
        """Make an LLM API call with proactive rate limiting and retry."""
        model = model or config.MAIN_LLM_MODEL

        # Use per-model budget (e.g. 70b has tighter TPM than 8b)
        budget = _get_budget(self.provider, model)

        # Estimate tokens for this request
        # Cap output estimate at 1500 — responses rarely max out the full budget
        est_prompt_tokens = (len(system_prompt) + len(user_prompt)) // 4
        est_output_tokens = min(max_tokens, 1500)
        est_total = est_prompt_tokens + est_output_tokens

        for attempt in range(config.MAX_RETRIES):
            # Proactive wait based on sliding window
            if budget:
                budget.wait_if_needed(est_total)

            try:
                try:
                    resp = self._call_litellm(system_prompt, user_prompt, model,
                                               temperature, max_tokens)
                except ImportError:
                    resp = self._call_direct(system_prompt, user_prompt, model,
                                              temperature, max_tokens)
                
                # Record actual usage (estimate: prompt + response)
                actual_tokens = est_prompt_tokens + len(resp or "") // 4
                if budget:
                    budget.record(actual_tokens)
                self._call_count += 1
                return resp

            except Exception as e:
                err_str = str(e).lower()
                # Check both string content AND exception class name for reliability
                exc_type = type(e).__name__.lower()
                is_rate_limit = ("rate_limit" in err_str or "rate limit" in err_str 
                                 or "429" in err_str or "ratelimiterror" in err_str
                                 or "ratelimit" in exc_type)
                is_overloaded = ("503" in err_str or "service unavailable" in err_str
                                 or "overloaded" in err_str or "high demand" in err_str
                                 or "serviceunavailable" in exc_type or "unavailable" in exc_type)
                is_daily_limit = ("tokens per day" in err_str or "tpd" in err_str
                                  or "daily" in err_str and "quota" in err_str)

                if is_daily_limit:
                    raise RuntimeError(
                        "Groq daily token quota exhausted. Wait until reset or "
                        "upgrade at console.groq.com/settings/billing"
                    ) from e

                if is_rate_limit or is_overloaded:
                    # Extract retry-after hint from error if present (e.g. Gemini "retry in 36s")
                    retry_match = re.search(r'retry[^\d]*(\d+)', err_str)
                    suggested = int(retry_match.group(1)) + 5 if retry_match else None
                    if suggested:
                        wait = suggested
                    elif is_overloaded:
                        wait = min(15 * (2 ** attempt), 60)  # 15→30→60s for transient overload
                    else:
                        wait = min(30 * (2 ** attempt), 120)  # 30→60→120s for rate limits
                    label = "Overloaded" if is_overloaded else "Rate limited"
                    print(f"  [{label}] Waiting {wait:.0f}s before retry "
                          f"{attempt+1}/{config.MAX_RETRIES}...")
                    time.sleep(wait)
                    if budget and is_rate_limit:
                        budget.record(est_total // 2)
                    continue
                raise

        raise RuntimeError(f"Rate limited after {config.MAX_RETRIES} retries")

    def _call_litellm(self, system_prompt, user_prompt, model,
                       temperature, max_tokens) -> str:
        """Use litellm for unified API."""
        import litellm
        
        model_str = self._litellm_model_string(model)
        
        # Large local models (32b+) on CPU can take 10-15min per turn.
        # Set timeout to 20min to cover curator + main LLM in HYBRID mode.
        timeout = 1200 if "ollama" in model_str else 120
        
        extra_kwargs = {}
        if "ollama" in model_str:
            # Ensure context window is large enough for long conversations
            extra_kwargs["num_ctx"] = config.OLLAMA_CONTEXT_SIZE
        
        response = litellm.completion(
            model=model_str,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **extra_kwargs,
        )
        return response.choices[0].message.content

    def _call_direct(self, system_prompt, user_prompt, model,
                      temperature, max_tokens) -> str:
        """Direct API calls without litellm."""
        if self.provider == "groq":
            return self._call_groq(system_prompt, user_prompt, model,
                                    temperature, max_tokens)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, model,
                                      temperature, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt, model,
                                      temperature, max_tokens)
        else:
            raise ValueError(f"Provider '{self.provider}' requires litellm. "
                           "Install with: pip install litellm")

    def _call_groq(self, system_prompt, user_prompt, model,
                    temperature, max_tokens) -> str:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _call_openai(self, system_prompt, user_prompt, model,
                      temperature, max_tokens) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _call_ollama(self, system_prompt, user_prompt, model,
                      temperature, max_tokens) -> str:
        import requests
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _litellm_model_string(self, model: str) -> str:
        """Convert to litellm format: provider/model."""
        prefix_map = {
            "groq": "groq/",
            "openai": "",
            "anthropic": "anthropic/",
            "google": "gemini/",
            "ollama": "ollama/",
        }
        prefix = prefix_map.get(self.provider, "")
        if model.startswith(prefix):
            return model
        return f"{prefix}{model}"

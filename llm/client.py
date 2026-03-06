"""
LLM Client — Multi-provider wrapper for LLM API calls.

Supports: Groq, OpenAI, Anthropic, Google, Ollama (local).
Uses litellm for unified API when available, falls back to direct calls.
Includes sliding-window rate limiter (modeled on llm-reasoning-pipeline).
"""

import time
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


def _get_budget(provider: str) -> Optional[TokenBudget]:
    """Get or create a shared TokenBudget for the provider."""
    if provider == "ollama":
        return None  # No rate limiting for local
    if provider not in _budgets:
        if provider == "groq":
            _budgets[provider] = TokenBudget(tpm_limit=12000, rpm_limit=30, min_gap=5.0)
        elif provider == "openai":
            _budgets[provider] = TokenBudget(tpm_limit=90000, rpm_limit=60, min_gap=1.0)
        elif provider == "google":
            # Gemini free tier: 15 RPM, 250k TPM → stay under 10 RPM for safety
            _budgets[provider] = TokenBudget(tpm_limit=250000, rpm_limit=10, min_gap=7.0)
        else:
            _budgets[provider] = TokenBudget(tpm_limit=60000, rpm_limit=60, min_gap=1.0)
    return _budgets[provider]


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
        self._budget = _get_budget(self.provider)

    def call(self, system_prompt: str, user_prompt: str,
             model: str = None, temperature: float = 0.3,
             max_tokens: int = 4096) -> str:
        """Make an LLM API call with proactive rate limiting and retry."""
        model = model or config.MAIN_LLM_MODEL

        # Estimate tokens for this request
        est_prompt_tokens = (len(system_prompt) + len(user_prompt)) // 4
        est_total = est_prompt_tokens + max_tokens

        for attempt in range(config.MAX_RETRIES):
            # Proactive wait based on sliding window
            if self._budget:
                self._budget.wait_if_needed(est_total)

            try:
                try:
                    resp = self._call_litellm(system_prompt, user_prompt, model,
                                               temperature, max_tokens)
                except ImportError:
                    resp = self._call_direct(system_prompt, user_prompt, model,
                                              temperature, max_tokens)
                
                # Record actual usage (estimate: prompt + response)
                actual_tokens = est_prompt_tokens + len(resp or "") // 4
                if self._budget:
                    self._budget.record(actual_tokens)
                self._call_count += 1
                return resp

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = ("rate_limit" in err_str or "rate limit" in err_str 
                                 or "429" in err_str)
                is_daily_limit = ("tokens per day" in err_str or "tpd" in err_str)
                
                if is_daily_limit:
                    raise RuntimeError(
                        "Groq daily token quota exhausted. Wait until reset or "
                        "upgrade at console.groq.com/settings/billing"
                    ) from e
                
                if is_rate_limit:
                    wait = min(20 * (2 ** attempt), 65)
                    print(f"  [Rate limited] Waiting {wait:.0f}s before retry "
                          f"{attempt+1}/{config.MAX_RETRIES}...")
                    time.sleep(wait)
                    # Record estimated tokens so the budget knows the window is hot
                    if self._budget:
                        self._budget.record(est_total // 2)
                    continue
                raise

        raise RuntimeError(f"Rate limited after {config.MAX_RETRIES} retries")

    def _call_litellm(self, system_prompt, user_prompt, model,
                       temperature, max_tokens) -> str:
        """Use litellm for unified API."""
        import litellm
        
        model_str = self._litellm_model_string(model)
        
        response = litellm.completion(
            model=model_str,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
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

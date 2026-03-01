"""
LLM Client — Multi-provider wrapper for LLM API calls.

Supports: Groq, OpenAI, Anthropic, Google, Ollama (local).
Uses litellm for unified API when available, falls back to direct calls.
"""

import time
import json
from typing import Optional

import config


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
        self._last_call_time = 0
        self._call_count = 0

    def call(self, system_prompt: str, user_prompt: str,
             model: str = None, temperature: float = 0.3,
             max_tokens: int = 2000) -> str:
        """
        Make an LLM API call.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            model: Model name (provider-specific)
            temperature: Sampling temperature
            max_tokens: Max response tokens
            
        Returns:
            The LLM's response text
        """
        model = model or config.MAIN_LLM_MODEL
        self._rate_limit()

        try:
            return self._call_litellm(system_prompt, user_prompt, model,
                                       temperature, max_tokens)
        except ImportError:
            return self._call_direct(system_prompt, user_prompt, model,
                                      temperature, max_tokens)

    def _call_litellm(self, system_prompt, user_prompt, model,
                       temperature, max_tokens) -> str:
        """Use litellm for unified API."""
        import litellm
        
        # Map provider to litellm model format
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

    def _rate_limit(self):
        """Simple rate limiting per provider."""
        delay = config.RATE_LIMIT_DELAY.get(self.provider, 0)
        elapsed = time.time() - self._last_call_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_call_time = time.time()
        self._call_count += 1

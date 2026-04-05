"""
Baseline 4: MemGPT-style self-managed memory.

The LLM itself decides what to store, retrieve, and forget
via function calls. Simulates the MemGPT approach.
"""

from typing import List, Optional
from dataclasses import dataclass

from llm.client import LLMClient
from llm.token_counter import count_tokens
from memory.vector_store import VectorStore
import config
import json


MEMGPT_SYSTEM_PROMPT = """You are an assistant with memory management capabilities.
You have access to a memory store. Before answering, decide if you need to:
1. SEARCH memory for relevant past context
2. STORE important information from this conversation

For each turn, first output a JSON action block, then your response:
{"action": "search", "query": "what to search for"}
or
{"action": "store", "content": "what to remember"}
or
{"action": "none"}

Then provide your response after a line containing only "---RESPONSE---"
"""


@dataclass
class MemGPTTurnResult:
    turn_number: int
    user_message: str
    assistant_response: str
    prompt_tokens: int
    response_tokens: int
    context_tokens: int
    memory_action: str


class MemGPTStyleBaseline:
    """
    MemGPT-style baseline: LLM manages its own memory via function calls.
    """

    def __init__(self, provider: str = None, model: str = None,
                 max_context_tokens: int = 6000):
        self.llm = LLMClient(provider=provider or config.DEFAULT_PROVIDER)
        self.model = model or config.MAIN_LLM_MODEL
        self.vector_store = VectorStore(collection_name="memgpt_baseline")
        self.max_context_tokens = max_context_tokens
        self.working_memory: List[str] = []  # Recent items kept in context
        self.max_working_memory = 5
        self.turn_count = 0

    def process_turn(self, user_message: str,
                     system_prompt: str = None) -> MemGPTTurnResult:
        self.turn_count += 1
        system_prompt = system_prompt or MEMGPT_SYSTEM_PROMPT

        # Build context with working memory
        context_parts = []
        if self.working_memory:
            context_parts.append("=== WORKING MEMORY ===")
            context_parts.extend(self.working_memory)
        context_parts.append(f"\n=== USER MESSAGE ===\n{user_message}")
        context = "\n".join(context_parts)

        # Call LLM (it decides memory actions + generates response)
        raw_response = self.llm.call(
            system_prompt=system_prompt,
            user_prompt=context,
            model=self.model,
            temperature=config.TEMPERATURE_MAIN,
        )

        # Parse action and response
        action, response = self._parse_response(raw_response)

        # Execute memory action
        self._execute_action(action, user_message, response)

        return MemGPTTurnResult(
            turn_number=self.turn_count,
            user_message=user_message,
            assistant_response=response,
            prompt_tokens=count_tokens(context),
            response_tokens=count_tokens(response),
            context_tokens=count_tokens(context),
            memory_action=action.get("action", "none") if isinstance(action, dict) else "none"
        )

    def _parse_response(self, raw: str) -> tuple:
        """Parse the LLM's combined action+response output."""
        action = {"action": "none"}
        response = raw

        if "---RESPONSE---" in raw:
            parts = raw.split("---RESPONSE---", 1)
            try:
                # Try to parse JSON action from first part
                action_text = parts[0].strip()
                if action_text.startswith("{"):
                    action = json.loads(action_text)
            except json.JSONDecodeError:
                pass
            response = parts[1].strip() if len(parts) > 1 else raw

        return action, response

    def _execute_action(self, action: dict, user_msg: str, response: str):
        """Execute the memory action decided by the LLM."""
        action_type = action.get("action", "none")

        if action_type == "search":
            query = action.get("query", user_msg)
            results = self.vector_store.query(query_text=query, n_results=3)
            # Add search results to working memory
            for r in results:
                text = r.get("text", "")[:200]
                if text and text not in self.working_memory:
                    self.working_memory.append(f"[Retrieved] {text}")

        if action_type == "store":
            content = action.get("content", "")
            if content:
                self.vector_store.add(text=content, metadata={"turn": self.turn_count})

        # Always store the turn
        self.vector_store.add(
            text=f"Turn {self.turn_count} - User: {user_msg[:200]} | Assistant: {response[:200]}",
            metadata={"turn": self.turn_count}
        )

        # Manage working memory size
        while len(self.working_memory) > self.max_working_memory:
            self.working_memory.pop(0)

    def reset(self):
        self.vector_store.clear()
        self.working_memory.clear()
        self.turn_count = 0

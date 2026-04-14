"""
HierMem Configuration

All system parameters in one place. Override via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
_default_data_root = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "HierMem"
HIERMEM_DATA_DIR = Path(os.getenv("HIERMEM_DATA_DIR", str(_default_data_root)))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", str(HIERMEM_DATA_DIR / "chroma_data")))
RESULTS_PATH = PROJECT_ROOT / "results"
PROMPTS_PATH = PROJECT_ROOT / "llm" / "prompts"
HIERMEM_STATE_DIR = Path(os.getenv("HIERMEM_STATE_DIR", str(HIERMEM_DATA_DIR / "state")))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hiermem")
CLEAR_VECTOR_ON_START = os.getenv("CLEAR_VECTOR_ON_START", "false").lower() in ("1", "true", "yes", "on")

# === LLM Provider Settings ===
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama")

# Per-role providers — set these to use different providers for different roles.
# If not set, falls back to DEFAULT_PROVIDER.
# Example: main on Groq (fast/strong), curator/summarizer on local ollama (free/unlimited)
MAIN_PROVIDER = os.getenv("MAIN_PROVIDER", DEFAULT_PROVIDER)
CURATOR_PROVIDER = os.getenv("CURATOR_PROVIDER", DEFAULT_PROVIDER)
SUMMARIZER_PROVIDER = os.getenv("SUMMARIZER_PROVIDER", CURATOR_PROVIDER)

# Model assignments per role
# For Ollama: use "ollama/model_name" format
# For Groq: "llama-3.1-70b-versatile", "llama-3.1-8b-instant"
MAIN_LLM_MODEL = os.getenv("MAIN_LLM_MODEL", "ollama/llama3.1:8b")
CURATOR_MODEL = os.getenv("CURATOR_MODEL", "ollama/qwen2.5:3b")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "ollama/qwen2.5:3b")

# Provider API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CONTEXT_SIZE = int(os.getenv("OLLAMA_CONTEXT_SIZE", "8192"))  # context window for local models
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")  # keep models resident to avoid reload thrash

# === Hierarchical Archive Settings ===
MAX_L0_ENTRIES = 20            # Max segments in topic directory
SEGMENT_SIZE = 10              # Turns per segment before creating new one
MAX_L1_TOKENS_PER_SEGMENT = 150  # Token budget per L1 segment summary
L0_MERGE_THRESHOLD = 25       # When L0 exceeds this, merge oldest segments

# === Constraint Store Settings ===
MAX_CONSTRAINT_TOKENS = 500    # Total token budget for constraint store
MAX_CONSTRAINTS = 20           # Max active constraints

# === Context Assembly Settings ===
# TOTAL_CONTEXT_BUDGET: Max tokens in assembled context for main LLM.
# 8192 matches MemGPT paper benchmark standard (~25% of qwen2.5:14b's 32K window).
# Groq TPM (6000) is UNRELATED — that rate-limits judge API calls, not main LLM context.
TOTAL_CONTEXT_BUDGET = int(os.getenv("TOTAL_CONTEXT_BUDGET", "8192"))
# Zone budgets — Z1 (constraints) and Z3 (peripheral) are fixed.
# Z4 (current prompt) scales with budget. Z2 (relevant context) gets the remainder.
ZONE_1_BUDGET = 700                                                             # Constraints zone (fixed, increased for stronger formatting)
ZONE_4_BUDGET = max(1200, TOTAL_CONTEXT_BUDGET // 5)                           # Current prompt zone (~20%)
ZONE_3_BUDGET = 300                                                             # Peripheral zone (fixed)
ZONE_2_BUDGET = TOTAL_CONTEXT_BUDGET - ZONE_1_BUDGET - ZONE_3_BUDGET - ZONE_4_BUDGET  # Relevant context (gets the rest)

# === Adaptive Passthrough ===
# When conversation history fits within this token threshold, use full history
# (like raw LLM) instead of curated context. Avoids information loss on short
# conversations. Memory archiving still runs in the background.
PASSTHROUGH_THRESHOLD = int(TOTAL_CONTEXT_BUDGET * 0.90)  # 7373 tokens at 8192 budget

# Per-system context budgets (for architecture experiments, e.g. Arch B/C)
HIERMEM_CONTEXT_BUDGET = int(os.getenv("HIERMEM_CONTEXT_BUDGET", str(TOTAL_CONTEXT_BUDGET)))
RAW_LLM_CONTEXT_BUDGET = int(os.getenv("RAW_LLM_CONTEXT_BUDGET", str(TOTAL_CONTEXT_BUDGET)))
RAG_CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", str(TOTAL_CONTEXT_BUDGET)))
RAG_SUMMARY_CONTEXT_BUDGET = int(os.getenv("RAG_SUMMARY_CONTEXT_BUDGET", str(TOTAL_CONTEXT_BUDGET)))

# Optional explicit passthrough threshold for HierMem; 0 => auto 90% of HierMem budget
HIERMEM_PASSTHROUGH_THRESHOLD = int(os.getenv("HIERMEM_PASSTHROUGH_THRESHOLD", "0"))

# === Curator Settings ===
MAX_SEGMENTS_TO_FETCH = 4      # Curator can select at most this many segments
MAX_SEMANTIC_QUERIES = 3       # Max vector search queries per turn
RECENT_TURNS_ALWAYS_INCLUDED = 3  # Always include last N turns

# === Retrieval Settings ===
SEMANTIC_TOP_K = 5             # Top-k results from vector search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-transformers model

# === Post-Processor Settings ===
ENABLE_CONSTRAINT_EXTRACTION = True
ENABLE_VIOLATION_CHECK = True
ENABLE_VIOLATION_RETRY = True        # Retry LLM call when constraint violation detected
ENABLE_AUTO_SUMMARIZE = True

# === LLM Call Settings ===
TEMPERATURE_MAIN = 0.3
TEMPERATURE_CURATOR = 0.0     # Deterministic curator
TEMPERATURE_SUMMARIZER = 0.0  # Deterministic summarization

MAX_TOKENS_MAIN = int(os.getenv("MAX_TOKENS_MAIN", "4096"))
MAX_TOKENS_CURATOR = int(os.getenv("MAX_TOKENS_CURATOR", "1024"))
MAX_TOKENS_SUMMARIZER = int(os.getenv("MAX_TOKENS_SUMMARIZER", "512"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RATE_LIMIT_DELAY = {
    "groq": 20.0,
    "openai": 0.5,
    "anthropic": 0.5,
    "google": 1.0,
    "ollama": 0.0,
}

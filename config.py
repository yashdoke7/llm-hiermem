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
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_data"
RESULTS_PATH = PROJECT_ROOT / "results"
PROMPTS_PATH = PROJECT_ROOT / "llm" / "prompts"

# === LLM Provider Settings ===
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama")

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

# === Hierarchical Archive Settings ===
MAX_L0_ENTRIES = 20            # Max segments in topic directory
SEGMENT_SIZE = 10              # Turns per segment before creating new one
MAX_L1_TOKENS_PER_SEGMENT = 150  # Token budget per L1 segment summary
L0_MERGE_THRESHOLD = 25       # When L0 exceeds this, merge oldest segments

# === Constraint Store Settings ===
MAX_CONSTRAINT_TOKENS = 500    # Total token budget for constraint store
MAX_CONSTRAINTS = 20           # Max active constraints

# === Context Assembly Settings ===
TOTAL_CONTEXT_BUDGET = 6000    # Max tokens in assembled context for main LLM
ZONE_1_BUDGET = 500            # Constraints zone
ZONE_2_BUDGET = 4000           # Relevant context zone
ZONE_3_BUDGET = 500            # Peripheral awareness zone
ZONE_4_BUDGET = 1000           # Current prompt zone (flexible)

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
ENABLE_AUTO_SUMMARIZE = True

# === LLM Call Settings ===
TEMPERATURE_MAIN = 0.3
TEMPERATURE_CURATOR = 0.0     # Deterministic curator
TEMPERATURE_SUMMARIZER = 0.0  # Deterministic summarization
MAX_RETRIES = 5
RATE_LIMIT_DELAY = {
    "groq": 20.0,
    "openai": 0.5,
    "anthropic": 0.5,
    "google": 1.0,
    "ollama": 0.0,
}

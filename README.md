<div align="center">

# 🧠 HierMem

### Hierarchical Paged Context Management for Long-Horizon LLM Reliability

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-hiermem-blue.svg)](https://pypi.org/project/hiermem/)
[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/paper-arXiv-red.svg)](docs/paper.tex)

*Inspired by OS virtual memory paging, HierMem manages LLM context through a multi-level hierarchical memory architecture with a stateless curator agent and priority-tagged constraint store — achieving significantly slower accuracy degradation over long conversations.*

[Quick Start](#quick-start) · [API Reference](#api-reference) · [Provider Setup](#provider-setup) · [Architecture](#architecture) · [Benchmarks](#evaluation) · [PyPI Deployment](#pypi-deployment)

</div>

---

## The Problem

Large Language Models degrade predictably as conversations grow:

- **Context Saturation** — After ~8-15K tokens, key facts and constraints get ignored
- **Constraint Forgetting** — Explicit rules stated early are violated later as they're pushed out of effective attention
- **Lost-in-the-Middle** ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)) — LLMs attend mostly to the beginning and end of context, losing information buried in the middle
- **O(n²) Cost Growth** — Full-history approaches cost exponentially more tokens as conversations lengthen

**Current approaches fail because** they treat memory as either (a) unlimited context window (overflows) or (b) flat retrieval over chunks (misses structure and priorities).

## Our Approach

HierMem draws from **OS virtual memory design** — the same principles that let your computer run programs larger than physical RAM:

| OS Concept | HierMem Analog |
|------------|----------------|
| Page Table | L0 Topic Directory — index of all memory segments |
| Page Frames | L1-L3 Levels — progressively detailed content |
| TLB (Translation Lookaside Buffer) | Constraint Store — always-loaded critical rules |
| Page Fault Handler | Curator Agent — decides what to load into context |
| Demand Paging | Adaptive Retrieval — fetch only what the query needs |
| Page Merging | L0 Segment Merging — compress old segments when index overflows |

---

## Quick Start

### Option 1: Install from PyPI (Recommended for Users)

```bash
pip install hiermem
```

### Option 2: Install from Source (For Development / Benchmarks)

```bash
git clone https://github.com/yashdoke7/llm-hiermem.git
cd llm-hiermem
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -e ".[dev,eval]"    # Editable install with all extras
```

### Minimal Example (5 lines)

```python
from core.pipeline import HierMemPipeline

# Create pipeline (uses .env or defaults to local Ollama)
pipeline = HierMemPipeline.create()

# Process conversation turns
result = pipeline.process_turn("Always respond in bullet points. What is Python?")
print(result.assistant_response)

# Constraints are automatically detected and persisted
result = pipeline.process_turn("Now explain JavaScript frameworks")
print(result.assistant_response)  # Still uses bullet points!
```

### With Explicit Provider Configuration

```python
import os
os.environ["MAIN_LLM_MODEL"] = "gpt-4o-mini"
os.environ["MAIN_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["CURATOR_MODEL"] = "ollama/qwen2.5:3b"  # Keep curator local (free)
os.environ["CURATOR_PROVIDER"] = "ollama"

from core.pipeline import HierMemPipeline
pipeline = HierMemPipeline.create()

# Or pass directly:
pipeline = HierMemPipeline.create(
    main_model="gpt-4o-mini",
    provider="openai",
    curator_model="ollama/qwen2.5:3b"
)
```

---

## API Reference

### Core Classes

#### `HierMemPipeline` — Main Entry Point

```python
from core.pipeline import HierMemPipeline

pipeline = HierMemPipeline.create(
    main_model=None,       # Override MAIN_LLM_MODEL (e.g. "gpt-4o-mini")
    curator_model=None,    # Override CURATOR_MODEL (e.g. "ollama/qwen2.5:3b")
    provider=None,         # Override MAIN_PROVIDER (e.g. "openai")
)
```

| Method | Description |
|--------|-------------|
| `create(main_model, curator_model, provider)` | Factory method. Initializes all components (vector store, constraint store, archive, curator, assembler, post-processor). |
| `process_turn(user_message, system_prompt)` | Process one conversation turn. Returns `TurnResult`. |

#### `TurnResult` — Response from Each Turn

```python
result = pipeline.process_turn("Your message here")

result.turn_number          # int — current turn index
result.user_message         # str — the input message
result.assistant_response   # str — the LLM's response
result.tokens_used          # int — total tokens in assembled context
result.curator_decision     # CuratorDecision — what retrieval strategy was used
result.assembled_context    # AssembledContext — the final prompt sent to LLM
result.warnings             # List[str] — any warnings (e.g., constraint violations)

# Telemetry (granular cost tracking)
result.pipeline_telemetry   # dict with keys:
  # passthrough_mode        — bool: True if full history fit under threshold
  # curator_input_tokens    — int: tokens the curator read
  # curator_output_tokens   — int: tokens the curator generated
  # postproc_summary_tokens — int: tokens used by summarizer
  # violation_retry_fired   — bool: True if constraint violation retry triggered
  # l0_entry_count          — int: segments in L0 directory
  # constraint_count        — int: active constraints
```

#### `ConstraintStore` — Access Active Constraints

```python
store = pipeline.constraint_store

store.get_all()             # List of all active constraints
store.get_display_text()    # Formatted string for prompt injection
store.count                 # Number of active constraints
```

#### `LLMClient` — Multi-Provider LLM Wrapper

```python
from llm.client import LLMClient

client = LLMClient(provider="openai")  # or "groq", "anthropic", "google", "ollama"
response = client.call(
    system_prompt="You are helpful.",
    user_prompt="Hello!",
    model="gpt-4o-mini",       # Model name for the provider
    temperature=0.3,           # 0.0 = deterministic, 1.0 = creative
    max_tokens=4096,           # Max response length
)
```

**Supported providers and models:**

| Provider | Example Models | Auth Required | Rate Limited |
|----------|---------------|---------------|--------------|
| `ollama` | `qwen2.5:14b`, `llama3.1:8b`, `qwen2.5:32b` | No (local) | No |
| `groq` | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | `GROQ_API_KEY` | Yes (free tier) |
| `openai` | `gpt-4o-mini`, `gpt-4o`, `gpt-4.1` | `OPENAI_API_KEY` | Yes |
| `anthropic` | `claude-3-5-sonnet`, `claude-3-haiku` | `ANTHROPIC_API_KEY` | Yes |
| `google` | `gemini-2.5-flash`, `gemini-1.5-flash` | `GOOGLE_API_KEY` | Yes (RPD limited) |

> **How it works:** `LLMClient` uses [litellm](https://docs.litellm.ai/) as a unified backend. If litellm isn't installed, it falls back to direct API calls for Groq, OpenAI, and Ollama. For Anthropic and Google, litellm is required.

#### `VectorStore` — ChromaDB Wrapper

```python
from memory.vector_store import VectorStore

store = VectorStore(collection_name="my_session", persist_dir="./chroma_data")
store.add("Some text", metadata={"turn": 5, "type": "summary"})
results = store.query("search query", n_results=5)
```

> Automatically falls back to in-memory keyword search if ChromaDB is not installed.

---

## Provider Setup

### Ollama (Free, Local — Recommended)

```bash
# 1. Install from https://ollama.com
# 2. Pull models:
ollama pull qwen2.5:14b    # Main LLM (~9GB) — best quality/speed
ollama pull qwen2.5:3b     # Curator/Summarizer (~2GB)

# 3. No .env needed — defaults point to Ollama
python -c "from core.pipeline import HierMemPipeline; p = HierMemPipeline.create(); print(p.process_turn('Hello!').assistant_response[:100])"
```

### OpenAI (GPT-4o, GPT-4o-mini)

```bash
# .env file:
MAIN_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
MAIN_LLM_MODEL=gpt-4o-mini
CURATOR_PROVIDER=ollama          # Keep curator local (free, no rate limits)
CURATOR_MODEL=ollama/qwen2.5:3b
SUMMARIZER_MODEL=ollama/qwen2.5:3b
```

### Anthropic (Claude)

```bash
# .env file:
MAIN_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
MAIN_LLM_MODEL=claude-3-5-sonnet-20241022
CURATOR_PROVIDER=ollama
CURATOR_MODEL=ollama/qwen2.5:3b
```

### Google Gemini (Free tier available)

```bash
# .env file:
MAIN_PROVIDER=google
GOOGLE_API_KEY=your-key-here
MAIN_LLM_MODEL=gemini/gemini-2.5-flash
CURATOR_PROVIDER=ollama
```

> ⚠️ Gemini free tier: 5 RPM, 20 RPD on Gemini 2.5 Flash. Use local Ollama for heavy workloads.

### Groq (Free tier, fast inference)

```bash
# .env file:
MAIN_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
MAIN_LLM_MODEL=llama-3.3-70b-versatile
CURATOR_PROVIDER=ollama
CURATOR_MODEL=ollama/qwen2.5:3b
```

### Mixed Providers (Best Practice)

You can mix providers per role — e.g., paid API for the main LLM, free local for curator/summarizer:

```python
# .env
MAIN_PROVIDER=openai               # Strong main LLM
MAIN_LLM_MODEL=gpt-4o-mini         # $0.15/$0.60 per 1M tokens
CURATOR_PROVIDER=ollama             # Free, unlimited curator
CURATOR_MODEL=ollama/qwen2.5:3b    # Fast, cheap, deterministic
SUMMARIZER_PROVIDER=ollama          # Free, unlimited summarizer
SUMMARIZER_MODEL=ollama/qwen2.5:3b
```

---

## Architecture

<div align="center">

![HierMem Architecture](docs/ArchitectureDiagram.png)

</div>

### Pipeline Flow

Every conversation turn goes through 5 stages:

```
1. CURATOR        Stateless LLM reads L0 index + constraints (~1000 tokens, constant)
                  → Decides retrieval strategy and which segments to fetch
                  
2. RETRIEVAL      Routes to KEYWORD / HIERARCHY / SEMANTIC / HYBRID / NONE
                  → Fetches relevant content from L1/L2/L3

3. ASSEMBLER      Builds attention-optimized context with 4 zones:
                  ┌─────────────────────────────────┐
                  │ Zone 1: Constraints (TOP)       │ ← highest attention
                  │ Recent Turns (last 3)           │
                  │ Zone 2: Retrieved Context       │
                  │ Zone 3: Peripheral Summaries    │ ← lowest attention
                  │ Zone 4: Current Prompt (BOTTOM) │ ← high attention
                  └─────────────────────────────────┘

4. MAIN LLM      Generates response from assembled context (~8192 tokens, configurable)

5. POST-PROCESS   Extracts new constraints, checks violations, summarizes turn,
                  updates archive (L0/L1/L2/L3) — ALL writes happen here
```

### Memory Hierarchy

| Level | What It Stores | Size | Access Pattern |
|-------|---------------|------|----------------|
| **L0** — Topic Directory | One-line segment labels | ~20 tokens/entry × 20 entries | Always in curator context |
| **L1** — Segment Summaries | Bullet-point summaries per segment | ~100 tokens/segment | Fetched selectively by curator |
| **L2** — Chunk Summaries | Per-turn condensed summaries | ~50 tokens/turn | ChromaDB semantic search |
| **L3** — Raw Content | Full original user + assistant text | ~200 tokens/turn | ChromaDB exact retrieval (rare) |
| **Constraints** | Priority-tagged rules & preferences | ~500 tokens max | Always included in main LLM context |

### Why It Doesn't Degrade

```
Curator input:  ~1000 tokens  (constant, regardless of conversation length)
Post-Processor: ~400 tokens   (processes one turn at a time)
Main LLM:      ~6000 tokens   (adaptive budget, never overflows)
```

Neither the curator nor the post-processor ever sees the full conversation history. They operate on **bounded, constant-size inputs** — the nesting problem (secondary LLM degrading) is architecturally impossible.

---

## Configuration

All parameters live in `config.py` and can be overridden via environment variables or `.env` file.

### Provider Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAIN_PROVIDER` | `ollama` | Provider for main LLM |
| `CURATOR_PROVIDER` | `ollama` | Provider for curator agent |
| `SUMMARIZER_PROVIDER` | `ollama` | Provider for summarizer |
| `MAIN_LLM_MODEL` | `ollama/llama3.1:8b` | Main LLM model |
| `CURATOR_MODEL` | `ollama/qwen2.5:3b` | Curator model (small, fast) |
| `SUMMARIZER_MODEL` | `ollama/qwen2.5:3b` | Summarizer model |

### API Keys

| Variable | Required For |
|----------|-------------|
| `GROQ_API_KEY` | Groq provider |
| `OPENAI_API_KEY` | OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `GOOGLE_API_KEY` | Google Gemini provider |
| `OLLAMA_BASE_URL` | Custom Ollama endpoint (default: `http://localhost:11434`) |

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_CONTEXT_BUDGET` | `8192` | Max tokens in assembled context |
| `HIERMEM_CONTEXT_BUDGET` | `8192` | HierMem-specific context budget (for arch experiments) |
| `HIERMEM_PASSTHROUGH_THRESHOLD` | `0` (auto=90%) | Token threshold before curation activates |
| `MAX_L0_ENTRIES` | `20` | Max segments in topic directory |
| `SEGMENT_SIZE` | `10` | Turns per segment before rotation |
| `MAX_CONSTRAINTS` | `20` | Max active constraints |
| `OLLAMA_CONTEXT_SIZE` | `8192` | Context window for local Ollama models |

### Post-Processor Toggles

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_CONSTRAINT_EXTRACTION` | `True` | Auto-detect constraints from user messages |
| `ENABLE_VIOLATION_CHECK` | `True` | Check responses for constraint violations |
| `ENABLE_VIOLATION_RETRY` | `True` | Retry LLM call when violation detected |
| `ENABLE_AUTO_SUMMARIZE` | `True` | Summarize turns for L1/L2 archiving |

---

## Project Structure

```
llm-hiermem/
├── config.py                          # All configuration (env-overridable)
├── pyproject.toml                     # PyPI package configuration
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
│
├── core/                              # Core pipeline modules
│   ├── pipeline.py                    # HierMemPipeline — main entry point
│   ├── curator.py                     # Stateless context selection agent
│   ├── assembler.py                   # Attention-aware 4-zone context assembly
│   ├── constraint_store.py            # Priority-tagged constraint store (TLB analog)
│   ├── archive.py                     # Hierarchical paged memory (L0-L3)
│   └── post_processor.py             # Post-response constraint violation + summarize
│
├── memory/                            # Storage backends
│   ├── vector_store.py                # ChromaDB wrapper (+ in-memory fallback)
│   ├── hierarchical_index.py          # L0/L1 index helper
│   └── embedding.py                   # Sentence-transformers wrapper
│
├── retrieval/                         # Retrieval strategies
│   ├── router.py                      # Routes curator decisions → strategies
│   ├── keyword.py                     # Keyword/regex search
│   ├── semantic.py                    # Vector similarity search
│   └── hierarchical.py               # L0 → L1 → L2 tree traversal
│
├── llm/                               # LLM integration
│   ├── client.py                      # Multi-provider client (5 providers)
│   ├── token_counter.py               # Token counting utilities
│   └── prompts/                       # System prompts for curator, summarizer
│
├── baselines/                         # 4 baseline systems for comparison
│   ├── raw_llm.py                     # Full history, truncate oldest
│   ├── rag_baseline.py                # ChromaDB top-k retrieval
│   ├── rag_summary.py                 # RAG + rolling conversation summary
│   └── memgpt_style.py               # LLM self-manages memory
│
├── eval/                              # Evaluation framework
│   ├── run_benchmark.py               # Benchmark runner (all systems, resume)
│   ├── research_metrics.py            # Publication-grade metrics + 9 chart types
│   └── datasets/                      # 20 benchmark datasets (4 domains)
│
├── docs/                              # Documentation
│   ├── paper.tex                      # LaTeX research paper (ACL format)
│   ├── ArchitectureDiagram.png        # System architecture diagram
│   └── research_paper_blueprint.md    # Research terminology guide
│
├── demo/                              # Streamlit interactive demo
├── tests/                             # Unit tests (28 tests)
└── results/                           # Benchmark results & charts
```

---

## Evaluation

### Baselines

| # | System | Strategy | Weakness |
|---|--------|----------|----------|
| 1 | **Raw LLM** | Full history in context, truncate oldest | O(n²) cost, constraint loss |
| 2 | **RAG** | ChromaDB top-k retrieval per query | Loses conversation structure |
| 3 | **RAG + Summary** | RAG + rolling conversation summary | Summary degrades over time |
| 4 | **MemGPT-style** | LLM self-manages memory via function calls | "Forgot to look" failures |
| 5 | **HierMem** (ours) | Hierarchical paging + curator + constraints | — |

### Metrics (9 Visualization Charts)

| Chart | File | What It Shows |
|-------|------|---------------|
| Degradation Curve | `degradation_curve.png` | LAAJ score at each checkpoint — shows when systems start forgetting |
| Cumulative Token Cost | `cumulative_token_cost.png` | O(1) flat scaling (HierMem) vs O(n²) growth (RawLLM) |
| Context Utilization Heatmap | `context_utilization_heatmap.png` | Turn × System heatmap of budget usage |
| Constraint Survival Bar | `constraint_survival.png` | % checkpoints where score ≥ 5.0 |
| Cost vs Quality Scatter | `cost_vs_quality.png` | Efficiency frontier across systems |
| Cost per Quality Point | `cost_per_quality.png` | Token cost ÷ LAAJ score (lower = better) |
| Curator Overhead Pie | `curator_overhead.png` | HierMem's Main vs Curator vs Embedding tokens |
| Latency Comparison | `latency_comparison.png` | Average seconds per turn |
| Token Composition | `token_composition.png` | Stacked bar of token types per system |

### Running Benchmarks

```bash
# Run all 4 systems on a specific dataset
python -m eval.run_benchmark --systems hiermem raw_llm rag rag_summary \
    --benchmark constraint_tracking \
    --run-dir results/raw/benchmarks/my_run \
    --convo dataset_se_01

# Generate metrics + charts (after adding laaj.json from GPT-4.1 judging)
python -m eval.research_metrics --run-dir results/raw/benchmarks/my_run

# Aggregate across all datasets in an architecture folder
python -m eval.research_metrics --aggregate results/raw/benchmarks/qwen14b_arch_c
```

### Current Results

| Rank | System | Avg LAAJ Score (Qwen2.5-14B) |
|------|--------|------------------------------|
| 1 | **HierMem** | **8.4 – 8.7** |
| 2 | Raw LLM | 7.6 – 8.2 |
| 3 | RAG + Summary | 5.9 – 7.4 |
| 4 | RAG | 5.4 – 7.5 |

> Full 20-dataset sweeps across Arch A (8K) and Arch C (32K) are queued for RunPod execution.

---

## Demo

Interactive Streamlit app for testing the pipeline in real-time:

```bash
pip install hiermem[demo]   # or: pip install streamlit
streamlit run demo/app.py
```

Features: live conversation, sidebar showing active constraints, segment count, memory state.

---

## PyPI Deployment

### For Maintainers: Publishing to PyPI

#### 1. Prerequisites

```bash
pip install build twine
```

#### 2. Build the Package

```bash
python -m build
# Creates dist/hiermem-0.1.0.tar.gz and dist/hiermem-0.1.0-py3-none-any.whl
```

#### 3. Test on TestPyPI First

```bash
twine upload --repository testpypi dist/*
# Test install: pip install --index-url https://test.pypi.org/simple/ hiermem
```

#### 4. Publish to PyPI

```bash
twine upload dist/*
# Username: __token__
# Password: pypi-your-api-token-here
```

#### 5. CI/CD (GitHub Actions)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

Add `PYPI_API_TOKEN` to your GitHub repo settings → Secrets and Variables → Actions.

#### 6. Version Bumping

Update version in `pyproject.toml`:
```toml
[project]
version = "0.1.1"  # bump for each release
```

Then create a GitHub release tag:
```bash
git tag v0.1.1
git push origin v0.1.1
# Create release on GitHub → CI publishes automatically
```

### Install Extras

```bash
pip install hiermem          # Core only (pipeline + LLM client)
pip install hiermem[eval]    # + matplotlib, pandas, numpy for benchmarks
pip install hiermem[dev]     # + pytest, ruff, black for development
pip install hiermem[demo]    # + streamlit for interactive demo
pip install hiermem[eval,dev]  # Multiple extras
```

---

## Testing

```bash
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_constraint_store.py -v
```

28 tests covering: constraint store, hierarchical archive, context assembler, retrieval router, evaluation metrics.

---

## Research Context

### Related Work

| Paper | Relation to HierMem |
|-------|-------------------|
| [MemGPT](https://arxiv.org/abs/2310.08560) (Packer et al., 2023) | LLM self-manages memory — we use a dedicated stateless curator instead |
| [Self-RAG](https://arxiv.org/abs/2310.11511) (Asai et al., 2023) | Selective retrieval — we add hierarchical structure |
| [RAPTOR](https://arxiv.org/abs/2401.18059) (Sarthi et al., 2024) | Tree-structured summarization — we add paging and constraints |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2023) | Attention position bias — we exploit this in context assembly |

### Research Paper

The full LaTeX paper is at [`docs/paper.tex`](docs/paper.tex) (ACL/NeurIPS format, 8 pages + references).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use HierMem in your research, please cite:

```bibtex
@software{hiermem2026,
  title={HierMem: Hierarchical Paged Context Management for Constraint-Preserving Long-Horizon LLM Conversations},
  author={Doke, Yash},
  year={2026},
  url={https://github.com/yashdoke7/llm-hiermem}
}
```

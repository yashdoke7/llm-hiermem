<div align="center">

# 🧠 HierMem

### Hierarchical Paged Context Management for Long-Horizon LLM Reliability

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#)

*Inspired by OS virtual memory paging, HierMem manages LLM context through a multi-level hierarchical memory architecture with a stateless curator agent and priority-tagged constraint store — achieving significantly slower accuracy degradation over long conversations.*

[Architecture](#architecture) · [Quick Start](#quick-start) · [Benchmarks](#evaluation) · [Blueprint](docs/HierMem_Blueprint.md) · [Demo](#demo)

</div>

---

## The Problem

Large Language Models degrade predictably as conversations grow:

- **Context Saturation** — After ~8-15K tokens, key facts and constraints get ignored
- **Constraint Forgetting** — Explicit rules stated early are violated later as they're pushed out of effective attention
- **Lost-in-the-Middle** ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)) — LLMs attend mostly to the beginning and end of context, losing information buried in the middle
- **Cascading Hallucination** — Once the model generates from incomplete context, each step compounds errors

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

4. MAIN LLM      Generates response from assembled context (~6000 tokens adaptive)

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

**Key insight:** L3 is write-heavy, read-light. Like a filesystem — stores everything, loads only what's needed.

### Why It Doesn't Degrade

```
Curator input:  ~1000 tokens  (constant, regardless of conversation length)
Post-Processor: ~400 tokens   (processes one turn at a time)
Main LLM:      ~6000 tokens   (adaptive budget, never overflows)
```

Neither the curator nor the post-processor ever sees the full conversation history. They operate on **bounded, constant-size inputs** — the nesting problem (secondary LLM degrading) is architecturally impossible.

## What's Novel

1. **Multi-level paged memory for LLMs** — no existing system uses OS-inspired hierarchical paging
2. **Stateless curator agent** — unlike MemGPT where the main LLM manages its own memory (prone to "forgot to look" failures)
3. **Always-checked constraint store** — no existing system has structural priority memory that's guaranteed present in every response
4. **Degradation curve as primary metric** — most papers test fixed-length tasks; we measure how accuracy decays over conversation length

## Quick Start

### Prerequisites

- Python 3.11+
- 8GB+ RAM (16GB recommended for local models)
- GPU with 6GB+ VRAM (optional, for local Ollama inference)
- At least one LLM provider:

| Provider | Cost | Models | Setup |
|----------|------|--------|-------|
| **Ollama** (local) | Free | Llama 3.1 8B, Qwen 2.5 3B | [ollama.com](https://ollama.com) |
| **Groq** (cloud) | Free tier | Llama 3.1 8B/70B | [console.groq.com](https://console.groq.com) |
| **OpenAI** | Paid | GPT-4o-mini | [platform.openai.com](https://platform.openai.com) |
| **Anthropic** | Paid | Claude Haiku | [console.anthropic.com](https://console.anthropic.com) |

### Installation

```bash
# Clone the repository
git clone https://github.com/yashdoke7/llm-hiermem.git
cd llm-hiermem

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your provider and API key
```

### Setup with Ollama (Free, Local)

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b      # Main LLM (~4.7GB)
ollama pull qwen2.5:3b       # Curator/Summarizer (~2GB)
```

Default config already points to Ollama. No API key needed.

### Setup with Groq (Free, Cloud, Faster)

```bash
# Get free API key from https://console.groq.com
# Edit .env:
DEFAULT_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
MAIN_LLM_MODEL=llama-3.1-70b-versatile
CURATOR_MODEL=llama-3.1-8b-instant
SUMMARIZER_MODEL=llama-3.1-8b-instant
```

### Verify Installation

```bash
# Run tests (no API key needed)
python -m pytest tests/ -v

# Quick pipeline test (needs LLM provider)
python -c "from core.pipeline import HierMemPipeline; p = HierMemPipeline.create(); r = p.process_turn('Remember: I prefer Python.'); print(r.assistant_response[:200]); print('Constraints:', p.constraint_store.get_display_text())"
```

## Project Structure

```
llm-hiermem/
├── config.py                          # All configuration in one place
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
│
├── core/                              # Core pipeline modules
│   ├── pipeline.py                    # Main orchestration loop
│   ├── curator.py                     # Stateless context selection agent
│   ├── assembler.py                   # Attention-aware 4-zone context assembly
│   ├── constraint_store.py            # Priority-tagged constraint store (TLB)
│   ├── archive.py                     # Hierarchical paged memory (L0-L3)
│   └── post_processor.py             # Post-response processing & memory updates
│
├── memory/                            # Storage backends
│   ├── vector_store.py                # ChromaDB wrapper with in-memory fallback
│   ├── hierarchical_index.py          # L0/L1 index helper
│   └── embedding.py                   # Sentence-transformers wrapper
│
├── retrieval/                         # Retrieval strategies
│   ├── router.py                      # Routes curator decisions to strategies
│   ├── keyword.py                     # Keyword/regex search
│   ├── semantic.py                    # Vector similarity search
│   └── hierarchical.py               # L0 → L1 → L2 tree traversal
│
├── llm/                               # LLM integration
│   ├── client.py                      # Multi-provider client (Groq/OpenAI/Anthropic/Ollama)
│   ├── token_counter.py               # Token counting utilities
│   └── prompts/                       # System prompts for curator, summarizer, etc.
│
├── baselines/                         # 4 baseline systems for comparison
│   ├── raw_llm.py                     # Baseline 1: Full history, truncate oldest
│   ├── rag_baseline.py                # Baseline 2: ChromaDB top-k retrieval
│   ├── rag_summary.py                 # Baseline 3: RAG + rolling summary
│   └── memgpt_style.py               # Baseline 4: LLM self-manages memory
│
├── eval/                              # Evaluation framework
│   ├── run_benchmark.py               # Benchmark runner for all systems
│   ├── metrics.py                     # CVR, task accuracy, degradation curves
│   ├── visualize.py                   # Matplotlib comparison charts
│   ├── datasets/                      # Generated test datasets
│   └── generators/                    # Dataset generation scripts
│
├── tests/                             # Unit tests (28 tests)
├── demo/                              # Streamlit interactive demo
├── docs/                              # Documentation & diagrams
│   ├── ArchitectureDiagram.png        # System architecture diagram
│   ├── HierMem_Blueprint.md           # Full research blueprint
│   └── HierMem_Blueprint.docx         # Blueprint (Word format)
└── results/                           # Benchmark results & charts
```

## Evaluation

### Baselines

| # | System | Strategy | Weakness |
|---|--------|----------|----------|
| 1 | **Raw LLM** | Full history in context, truncate oldest | Context overflow at ~30-50 turns |
| 2 | **RAG** | ChromaDB top-k retrieval per query | Loses conversation structure |
| 3 | **RAG + Summary** | RAG + rolling conversation summary | Summary degrades over time |
| 4 | **MemGPT-style** | LLM self-manages memory via function calls | "Forgot to look" failures |
| 5 | **HierMem** (ours) | Hierarchical paging + curator + constraints | — |

### Benchmarks

| Benchmark | What It Tests | Dataset |
|-----------|--------------|---------|
| **Constraint Tracking** | Are rules stated early still followed 40 turns later? | 50 synthetic conversations × 50 turns |
| **Needle-in-a-Haystack** | Can a specific fact be recalled at varying distances? | 30 conversations with planted facts |
| **Multi-step Reasoning** | Can the system chain facts from different turns? | 40 conversations with dependent setup turns |

### Metrics

| Metric | Description |
|--------|-------------|
| **Degradation Curve** (primary) | Accuracy plotted as a function of conversation length |
| **Constraint Violation Rate (CVR)** | % of responses that violate stated constraints |
| **Fact Recall@K** | Recall accuracy at K turns distance from the stated fact |
| **Task Accuracy** | Overall correctness at checkpoint evaluations |

### Running Benchmarks

```bash
# Generate datasets (already included, but regeneratable)
python -m eval.generators.generate_constraint_convos
python -m eval.generators.generate_niah
python -m eval.generators.generate_multistep

# Run all systems on a benchmark
python -m eval.run_benchmark --systems all --benchmark constraint_tracking

# Run specific systems
python -m eval.run_benchmark --systems hiermem raw_llm --benchmark needle_in_haystack

# Generate comparison charts
python -m eval.visualize --results results/eval_YYYYMMDD_HHMMSS/
```

## Demo

Interactive Streamlit app for testing the pipeline in real-time:

```bash
streamlit run demo/app.py
```

Features:
- Live conversation with HierMem pipeline
- Sidebar showing: active constraints, segment count, turn count
- Real-time memory state visualization

## Configuration

All parameters are in `config.py` and can be overridden via environment variables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_CONTEXT_BUDGET` | 6000 | Max tokens in assembled context |
| `MAX_L0_ENTRIES` | 20 | Max segments in topic directory |
| `SEGMENT_SIZE` | 10 | Turns per segment before rotation |
| `MAX_CONSTRAINTS` | 20 | Max active constraints |
| `RECENT_TURNS_ALWAYS_INCLUDED` | 3 | Recent turns always in context |
| `SEMANTIC_TOP_K` | 5 | Top-k results from vector search |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_constraint_store.py -v
```

28 tests covering: constraint store, hierarchical archive, context assembler, retrieval router, evaluation metrics.

## Research Context

### Related Work

| Paper | Relation to HierMem |
|-------|-------------------|
| [MemGPT](https://arxiv.org/abs/2310.08560) (Packer et al., 2023) | LLM self-manages memory — we use a dedicated stateless curator instead |
| [Self-RAG](https://arxiv.org/abs/2310.11511) (Asai et al., 2023) | Selective retrieval — we add hierarchical structure |
| [RAPTOR](https://arxiv.org/abs/2401.18059) (Sarthi et al., 2024) | Tree-structured summarization — we add paging and constraints |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2023) | Attention position bias — we exploit this in context assembly |
| [LLMs Cannot Self-Correct](https://arxiv.org/abs/2310.01798) (Huang et al., 2024) | Self-correction needs external grounding — motivates our constraint store |

### Key Research Question

> *Can a hierarchical paged memory architecture with a dedicated curator agent and priority-tagged constraint store significantly delay accuracy degradation compared to raw LLM, standard RAG, and self-managed memory (MemGPT-style)?*

## Tech Stack

- **Python 3.11+** — Core language
- **ChromaDB** — Vector storage for L2/L3
- **sentence-transformers** — Local embeddings (`all-MiniLM-L6-v2`)
- **litellm** — Unified LLM API across providers
- **Pydantic** — Data validation
- **Streamlit** — Interactive demo
- **matplotlib/seaborn** — Evaluation visualization
- **pytest** — Testing framework

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use HierMem in your research, please cite:

```bibtex
@software{hiermem2026,
  title={HierMem: Hierarchical Paged Context Management for Long-Horizon LLM Reliability},
  year={2026},
  url={https://github.com/yashdoke7/llm-hiermem}
}
```

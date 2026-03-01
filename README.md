# HierMem: Hierarchical Paged Context Management for Long-Horizon LLM Reliability

> *Inspired by OS virtual memory, HierMem manages LLM context through a multi-level page table of hierarchical summaries, with a dedicated constraint store and stateless curator agent, achieving exponentially slower accuracy degradation over long conversations.*

## Quick Start

```bash
# 1. Clone and enter project
cd C:\Users\ASUS\Desktop\CS\Projects\hiermem

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
copy .env.example .env
# Edit .env with your API key(s)

# 5. (Optional) Install local models via Ollama
# Download from https://ollama.ai then:
ollama pull llama3.1:8b
ollama pull qwen2.5:3b

# 6. Run tests
pytest tests/ -v

# 7. Run a conversation
python -m core.pipeline --turns 20

# 8. Run evaluation
python eval/run_benchmark.py --systems all --benchmark constraint_convos

# 9. Generate charts
python eval/visualize.py --results results/latest/
```

## Architecture

```
User Prompt → Curator Agent → [Constraint Store + Hierarchical Archive + Vector DB]
                                              ↓
                                    Assembled Context (4 zones)
                                              ↓
                                         Main LLM
                                              ↓
                                      Post-Processor
                                    (extract constraints,
                                     summarize, archive)
```

### Memory Levels (Paging)
| Level | Content | Size | Who Sees It |
|-------|---------|------|-------------|
| L0 | Topic directory (1-line segment labels) | ~20 tokens/entry | Curator always |
| L1 | Segment summaries (bullet points) | ~100 tokens/seg | Curator fetches selectively |
| L2 | Turn summaries (per-turn condensed) | ~50-100 tokens | ChromaDB semantic search |
| L3 | Raw content (full original text) | Variable | ChromaDB exact retrieval |
| Constraints | Tagged rules/preferences | ~200 tokens total | Always included for both |

## Project Structure

```
hiermem/
├── config.py                  # All configuration
├── core/                      # Core system modules
│   ├── constraint_store.py    # Always-checked rules store
│   ├── archive.py             # Hierarchical paging archive
│   ├── curator.py             # Context selection agent
│   ├── assembler.py           # Attention-aware context assembly
│   ├── post_processor.py      # Post-response processing
│   └── pipeline.py            # Main orchestration
├── memory/                    # Storage backends
├── retrieval/                 # Retrieval strategies
├── llm/                       # LLM client & prompts
├── baselines/                 # 4 baseline systems
├── eval/                      # Benchmarks & evaluation
├── tests/                     # Unit & integration tests
└── demo/                      # Streamlit demo
```

## Evaluation

Run against 4 baselines (Raw LLM, RAG, RAG+Summary, MemGPT-style) on:
- Custom multi-turn constraint conversations (50 × 50 turns)
- Needle-in-a-Haystack (adapted for conversation)
- BABILong benchmark
- LooGLE benchmark

Primary metric: **Degradation Curve** — accuracy as function of conversation length.

## Prerequisites

- Python 3.11+
- 8GB+ RAM (16GB recommended for local models)
- (Optional) GPU with 8GB+ VRAM for local model inference
- (Optional) Ollama for local models
- At least one API key: Groq (free), OpenAI, Anthropic, or Google

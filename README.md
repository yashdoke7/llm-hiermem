<div align="center">

  <h1>🧠 HierMem</h1>

  **Hierarchical context management for long-horizon LLM conversations with explicit constraint preservation.**

  [![PyPI - Version](https://img.shields.io/pypi/v/hiermem.svg)](https://pypi.org/project/hiermem/)
  [![GitHub Release](https://img.shields.io/github/v/release/yashdoke7/llm-hiermem?color=success&label=Release)](https://github.com/yashdoke7/llm-hiermem/releases/latest)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  <a href="https://linkedin.com/in/yashdoke"><img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=Linkedin&logoColor=white" alt="LinkedIn"></a>

</div>

## 📌 Release Summary (v1.0.3)

The v1.0.3 release marks our transition to a stable, research-grade python package available on PyPI while solidifying the architecture for the upcoming research paper publication. It introduces:
- **Full multi-provider support** (Ollama, OpenAI, Anthropic, Google Gemini, Groq) via integrated LiteLLM routing.
- **Rigorous constraint-adherence benchmarking**, proving efficiency gains over standard RAG and baseline LLM techniques.
- **Dynamic Context Pacing**, automatically balancing L0-L3 memory access via the underlying curator model.
- **Context Pressure Tracking**, visually proving an overall **4.7x architectural compression ratio** on conversation context management.

---

## 📖 Table of Contents
- [Why HierMem?](#-why-hiermem)
- [Installation Guide ***(Crucial)***](#-installation-guide)
- [Quick Start](#-quick-start)
- [Setup & Configuration](#️-setup--configuration)
  - [API Keys & Variables](#api-keys--environment-variables)
  - [Deployment Details](#deployment-details)
- [Images & Dimension Setup](#-images--dimension-setup)
- [Benchmark Snapshot](#-benchmark-snapshot)
- [Research & Paper](#-research--paper)

---

## ⚡ Why HierMem?

Long conversations typically degrade due to "catastrophic forgetting", burying critical rules and logic. **HierMem** tackles this natively at the system architecture layer:

- **Constraint-first Protection:** Active rules have designated zones and are strictly protected against context overflow.
- **Four-Level Memory Hierarchy:**
  - **L0:** Topic Index
  - **L1:** Summaries
  - **L2:** Embeddings
  - **L3:** Raw Turns
- **Curator Orchestration:** Uses a smaller, efficient underlying model (e.g. Qwen2.5 3B) to select information and dynamically adapt the pipeline, significantly reducing compute costs in the main inference model.

---

## 🛠️ Installation Guide

> **⚠️ CRITICAL: Virtual Environment Required**
>
> To avoid dependency clashes with your global Python environment, you **MUST** create a virtual environment (`venv`) before installing.

### Standard Installation

```bash
# 1. Clone the repository
git clone https://github.com/yashdoke7/llm-hiermem.git
cd llm-hiermem

# 2. Create and activate a virtual environment (Mandatory)
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install the package
pip install hiermem
# ...or install from local source:
pip install -e .
```

### Optional Extras
If you want to use the benchmarking suite, development utilities, or streamlit UI demo, install the respective extras:
```bash
pip install -e .[eval]
pip install -e .[dev]
pip install -e .[demo]
```

---

## 🚀 Quick Start

Once installed, usage is straightforward in your custom Python apps.

**With Python API:**
```python
from core.pipeline import HierMemPipeline

# Automatically initiates the pipeline based on `.env` configuration
pipeline = HierMemPipeline.create()

# Interaction 1
r1 = pipeline.process_turn("Always answer in bullet points. What is Python?")
print(r1.assistant_response)

# Interaction 2
r2 = pipeline.process_turn("Now compare Python and Go for backend systems.")
print(r2.assistant_response)
```

**With the built-in CLI Tool:**
```bash
hiermem config
hiermem chat
hiermem ask "Give me a 3-point summary of memory hierarchies"
```

---

## ⚙️ Setup & Configuration

For optimal setup, define your configurations via `.env` or Environment Variables. 

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

### API Keys & Environment Variables

Only apply keys for providers you actually intend to use for your models:
```env
# Credentials
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Provider Routing
DEFAULT_PROVIDER=ollama
MAIN_PROVIDER=openai
CURATOR_PROVIDER=ollama
SUMMARIZER_PROVIDER=ollama

# Model Assignments
MAIN_LLM_MODEL=gpt-4o-mini
CURATOR_MODEL=ollama/qwen2.5:3b
SUMMARIZER_MODEL=ollama/qwen2.5:3b
```

### Deployment Details

HierMem functions as a **memory-orchestration runtime library**. It manages memory storage entirely locally (vector DB via `ChromaDB`), querying local or cloud inference providers simultaneously. 
*Note: Vector storage reset is set to `CLEAR_VECTOR_ON_START=false` for persistence by default. If running evaluation benchmarks, set this to `true` to ensure clean context blocks!*

---

## 🖼️ Images & Dimension Setup

> **Note for Contributors:**
> When adding layout and diagram images to the repository (especially for academic or PyPI use), it is highly recommended to stick to standard ratios to prevent scaling blur.
> - **Recommended Ratio:** `16:9`
> - **Recommended Dimensions:** `1920x1080` (or `1280x720` for lighter files)
> - **Hosting:** Images in this README have been deliberately hyperlinked via absolute raw GitHub URLs instead of relative paths, ensuring complete visibility on PyPI deployment pages.

---

## 📊 Benchmark Snapshot (Qwen2.5-14B)

HierMem achieves profound efficiency and reliability enhancements, validated by our Gemini 3.1 Pro continuous tracking evaluation across 15 synthetic datasets. 

| Metric | HierMem | Raw LLM | Delta |
|---|---:|---:|---:|
| **Mean Judge Score** (out of 10) | **8.461** | 6.908 | +1.553 |
| **Constraint Survival** | **0.933** | 0.740 | +0.193 |
| **Mean Compute Cost/Turn** | **$0.0176** | $0.0264 | -33.3% |
| **Compression Ratio** | **4.7x** | 3.2x (Truncated) | +46.8% |

### Evaluation Metrics Breakdown

**Context Pressure \& Compression Ratio**  
<img src="https://raw.githubusercontent.com/yashdoke7/llm-hiermem/main/docs/figures/overall_context_pressure.png" alt="Context Pressure Trend" width="100%"/>

**Overall Quality Trend**  
<img src="https://raw.githubusercontent.com/yashdoke7/llm-hiermem/main/docs/figures/overall_quality.png" alt="Overall Quality Trend" width="100%"/>

**Overall Pareto Frontier**  
<img src="https://raw.githubusercontent.com/yashdoke7/llm-hiermem/main/docs/figures/overall_pareto.png" alt="Overall Pareto Frontier" width="100%"/>

**Cost Breakdown**  
<img src="https://raw.githubusercontent.com/yashdoke7/llm-hiermem/main/docs/figures/overall_cost_breakdown.png" alt="Cost Breakdown" width="100%"/>

**Latency Trend**  
<img src="https://raw.githubusercontent.com/yashdoke7/llm-hiermem/main/docs/figures/overall_latency_trend.png" alt="Overall Latency Trend" width="100%"/>

---

## 🎓 Research & Paper

Our rigorous synthetic benchmarks reveal strong architectural integrity against memory decay in lengthy contexts. Find the draft manuscript in the repository under `docs/paper.tex`.

- **Release Paper Snapshot:** [v1.0.0-paper Release Tag](https://github.com/yashdoke7/llm-hiermem/releases/tag/v1.0.0-paper)
- **Dataset Access:** [HF Datasets: hiermem-constraint-tracking](https://huggingface.co/datasets/yashdoke7/hiermem-constraint-tracking)

### Citation
```bibtex
@software{doke2026hiermem,
  title={HierMem: Constraint-Preserving Hierarchical Context Management for Long-Horizon LLM Conversations},
  author={Yash Doke},
  year={2026},
  url={https://github.com/yashdoke7/llm-hiermem}
}
```

---
*Maintained by Yash Doke • [LinkedIn](https://linkedin.com/in/yashdoke)*

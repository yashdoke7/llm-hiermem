# HierMem Paper Evidence Notes (Grounded)

## 1) What the code actually does

### 1.1 Runtime orchestration (per turn)
- Source: core/pipeline.py
- Turn flow implemented in code:
  1. Curator decision (except early passthrough)
  2. Retrieval routing (L1/L2/L3)
  3. Attention-aware context assembly
  4. Main LLM call
  5. Post-processing (violation check, extraction, summary, archive updates)
- Passthrough behavior:
  - If estimated history tokens < passthrough threshold, full-history mode is used.
  - Threshold defaults to 90% of HierMem context budget unless explicitly set.

### 1.2 Memory hierarchy implementation
- Source: core/archive.py
- L0: topic directory (segment labels and ranges)
- L1: per-segment bullet summaries
- L2: semantic summaries in vector store
- L3: raw turn content in vector store
- Segment management:
  - New segment every SEGMENT_SIZE turns
  - Oldest L0 segments can be merged when L0 over budget

### 1.3 Curator implementation details
- Source: core/curator.py
- Curator is stateless by design.
- Retrieval strategy outputs:
  - KEYWORD, HIERARCHY, SEMANTIC, HYBRID, NONE
- Curator inputs:
  - Current prompt
  - L0 topic directory text
  - Active constraints
- Curator outputs include:
  - segments_to_fetch
  - semantic_queries
  - peripheral_segments
  - fetch_full_turns

### 1.4 Context assembly logic
- Source: core/assembler.py
- Zone ordering in assembled prompt:
  - Constraints (top)
  - Recent conversation
  - Relevant context
  - Peripheral summaries
  - Current request (bottom)
- Truncation priority:
  - Drop peripheral first, then lower-priority relevant chunks
  - Preserve constraints and current request

### 1.5 Post-processing and correction loop
- Source: core/post_processor.py
- After response:
  - LLM-based violation check against active constraints
  - Optional retry with violation feedback
  - Constraint extraction from user message
  - L1/L2 summarization + archive update

### 1.6 Main LLM calls and MCP clarification
- Source: llm/client.py and config.py
- System uses provider API calls via LLM client wrapper (litellm/direct adapters).
- Providers supported in code: Ollama, OpenAI, Groq, Google, Anthropic.
- No MCP-based execution path is used in the project runtime code.

## 2) Why this architecture is motivated (for paper narrative)

### 2.1 Problem mechanism
- Full-history systems can exceed practical budget and bury constraints inside long context.
- Pure RAG can retrieve relevant text but does not guarantee instruction salience.
- Rolling summaries can alter strict wording over long horizon.

### 2.2 Why curator + hierarchy are needed
- Curator offloads retrieval policy decisions from the expensive main model.
- Curator sees compact signals (L0 + constraints), reducing selection-token overhead.
- Hierarchical storage avoids repeatedly sending raw history.
- Constraint zone gives deterministic visibility independent of similarity retrieval.

### 2.3 Why a smaller curator (3B class) is practical
- Main-model calls dominate cost.
- Small model is sufficient for routing/retrieval decisions.
- Post-processor and curator overhead can still be lower than repeated large-context main calls.

## 3) Benchmark facts (Architecture C, qwen14b_arch_c)

### 3.1 Overall aggregate (15 datasets)
- Source: results/raw/benchmarks/qwen14b_arch_c/arch_metrics_research.json
- Mean judge score:
  - HierMem: 8.4613
  - Raw LLM: 6.9083
  - RAG: 6.4387
  - RAG Summary: 7.1477
- Mean compute cost per turn:
  - HierMem: 0.01762668
  - Raw LLM: 0.02643456
  - RAG: 0.0241514
  - RAG Summary: 0.02498797

### 3.2 Domain-level aggregates (computed from JSON)
- Domains: software engineering, finance, legal, healthcare.
- Finance:
  - HierMem best on quality, survival, and cost.
- Software engineering:
  - HierMem best on quality, survival, and cost.
- Healthcare:
  - HierMem best on quality, survival, and cost.
- Legal:
  - Raw LLM slightly higher quality/survival; HierMem still best cost.
- Interpretation:
  - Strong cross-domain trend in favor of HierMem with one domain where full-history remains competitive on quality.

### 3.3 Domain aggregate figures created
- docs/figures/domain_aggregate_quality.png
- docs/figures/domain_aggregate_survival.png
- docs/figures/domain_aggregate_pareto.png

## 4) What to show in paper (chart strategy)

### 4.1 Primary (main paper body)
- 4-domain aggregate quality chart
- 4-domain aggregate survival chart
- 4-domain aggregate cost-quality chart

### 4.2 Secondary (diagnostics / appendix)
- Representative single-dataset trend curves
- Failure-case examples where retrieval or summary drift appears

### 4.3 Why this is better than only single examples
- Reduces selection bias.
- Better supports cross-domain robustness claims.
- Aligns with standard research reporting expectations.

## 5) Related papers: what they typically present and how to align

### 5.1 RAG (Lewis et al. 2020)
- Core idea: combine parametric model with external non-parametric memory.
- Typical evidence style:
  - Task performance tables across QA/generation datasets
  - Comparisons versus parametric-only baselines
- For our alignment:
  - Keep strong baseline tables and quality/factuality comparisons.

### 5.2 Lost in the Middle (Liu et al. 2023)
- Core finding: performance depends on where relevant info appears in context.
- Typical evidence style:
  - Position-sensitivity curves
  - Multi-setting figures highlighting middle-position degradation
- For our alignment:
  - Explicitly motivate constraint-first zone placement with this phenomenon.

### 5.3 MemGPT (Packer et al. 2023)
- Core idea: OS-inspired virtual context and memory-tier movement.
- Typical evidence style:
  - Long-horizon task demos + system design framing
- For our alignment:
  - Emphasize difference: dedicated small curator model and explicit constraint-zone enforcement.

### 5.4 Self-RAG (Asai et al. 2023)
- Core idea: train model to decide when to retrieve and critique itself.
- Typical evidence style:
  - Many benchmark tables and ablation-heavy evaluation
- For our alignment:
  - Discuss why HierMem uses systems-level modular control instead of end-to-end retriever fine-tuning.

## 6) Manuscript decisions to apply now
- Use single-column format for readability and larger figures/tables.
- Keep one concise aggregate table (overall).
- Promote all-4-domain aggregate figures to main results.
- Keep single-dataset charts as diagnostic examples, not headline evidence.
- Add stronger method explanation of:
  - API-driven multi-provider calls (not MCP runtime)
  - Why full-history degrades
  - Why RAG alone is insufficient for strict constraints
  - Why small curator + hierarchical paging lowers effective cost
- Add explicit legal-domain caveat and analysis.

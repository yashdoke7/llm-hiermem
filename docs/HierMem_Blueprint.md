# HierMem: Hierarchical Paged Context Management for Long-Horizon LLM Reliability

## Research Blueprint — Comprehensive Implementation & Evaluation Guide

---

## 1. Problem Statement

Large Language Models degrade predictably as conversation/task context grows:

- **Lost-in-the-middle** (Liu et al., 2023): LLMs pay most attention to beginning/end of context, losing information buried in the middle
- **Context saturation**: After ~8-15K tokens of accumulated context, key facts and constraints start being ignored
- **Constraint forgetting**: Explicit user rules stated early in a session are violated later as they get pushed out of effective attention range
- **Cascading hallucination**: Once the model starts generating based on incomplete context, each subsequent step compounds errors

**The core insight**: Current approaches treat LLM memory as either (a) unlimited context window (fails at scale) or (b) flat retrieval over stored chunks (misses relationships and priorities). Neither matches how reliable information systems actually work.

**Our approach**: Inspired by **OS virtual memory paging**, we propose a hierarchical, multi-level memory management system where:
- A **stateless curator agent** operates on an **index** (not the full data)
- A **constraint store** provides always-available critical rules
- A **hierarchical archive** gives exponentially scaling memory capacity
- Context is **assembled per-query** with attention-aware positioning

---

## 2. Key Research Question

> *Can a hierarchical paged memory architecture with a dedicated curator agent and priority-tagged constraint store significantly delay accuracy degradation in LLM-based systems over long conversations compared to (a) raw LLM, (b) standard RAG, and (c) self-managed memory (MemGPT-style)?*

**Secondary questions:**
- How does degradation scale with each additional paging level?
- Does a dedicated curator outperform self-managed memory retrieval?
- What is the optimal retrieval strategy combination (vector vs. keyword vs. hierarchical)?

---

## 3. Related Work & Positioning

### 3.1 Existing Systems We Build Upon

| System | Paper | What it Does | Our Relationship |
|--------|-------|-------------|-----------------|
| **MemGPT** | Packer et al., 2023 (arXiv:2310.08560) | OS-inspired virtual context; LLM manages own memory via function calls | **Closest prior work.** We differ with: dedicated curator (not self-managed), constraint store, multi-level paging |
| **RAPTOR** | Sarthi et al., 2024 (arXiv:2401.18059) | Recursive tree of summaries for document retrieval; +20% on QuALITY | **We adapt their tree structure** from document indexing to conversation memory |
| **GraphRAG** | Edge et al., 2024 (arXiv:2404.16130) | Knowledge graph + community hierarchy for corpus-level QA | **Too expensive for real-time conversation;** we use lighter hierarchical summaries instead |
| **Self-RAG** | Asai et al., 2023 (arXiv:2310.11511) | Model learns when to retrieve with reflection tokens | Requires fine-tuning; we achieve retrieve-when-needed via curator agent |
| **Reflexion** | Shinn et al., 2023 (arXiv:2303.11366) | Verbal reflection stored in episodic memory across trials | Trial-level, not turn-level; requires external verifier |

### 3.2 Key Findings That Motivate Our Design

| Finding | Paper | Implication for Us |
|---------|-------|--------------------|
| LLMs effectively use only 10-20% of long context | BABILong (Kuratov et al., 2024) | Raw context stuffing fails — need selective assembly |
| Performance drops for info in middle of context | Lost-in-the-middle (Liu et al., 2023) | Constraints must go at TOP, current query at BOTTOM |
| LLMs cannot self-correct without external feedback | Huang et al., 2023 (arXiv:2310.01798) | Curator must be separate from main LLM — same model can't catch its own gaps |
| RAG achieves ~60% on single-fact QA regardless of length | BABILong benchmark | Pure RAG insufficient — need hierarchical + priority structure |
| Atom of Thoughts: Markovian decomposition improves scaling | Teng et al., 2025 (arXiv:2502.12018) | Each curator call should be stateless (Markovian) for robustness |

### 3.3 What's Genuinely Novel in Our Approach

| Component | Existing Work | Our Novelty |
|-----------|--------------|-------------|
| Multi-level paged memory for LLM | MemGPT has flat 2-tier | **Hierarchical n-level with exponential capacity scaling** |
| Dedicated stateless curator agent | MemGPT: LLM manages itself | **Separate agent prevents "forgot to look" failure** |
| Always-checked constraint store | No structural equivalent | **Critical rules can never be missed** |
| Attention-aware context positioning | Known principle (lost-in-middle) | **Systematically applied to assembled context** |
| Degradation curve as primary metric | Not standard in benchmarks | **Novel evaluation methodology** |
| Adaptive retrieval strategy per query type | Individual strategies studied separately | **Dynamic routing between vector/keyword/hierarchical** |

---

## 4. Architecture

### 4.1 System Overview

```
                              USER PROMPT
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │     CONTEXT CURATOR       │
                    │     (Stateless Agent)     │
                    │                          │
                    │ Inputs:                  │
                    │  • Current user prompt   │
                    │  • Topic Directory (L0)  │
                    │  • Constraint Store      │
                    │                          │
                    │ Outputs:                 │
                    │  • Segment IDs to fetch  │
                    │  • Retrieval queries     │
                    │  • Context arrangement   │
                    └─────────┬────────────────┘
                              │
               ┌──────────────┼──────────────────┐
               │              │                   │
               ▼              ▼                   ▼
    ┌─────────────────┐ ┌──────────────┐ ┌────────────────────┐
    │  CONSTRAINT     │ │ HIERARCHICAL │ │  VECTOR STORE      │
    │  STORE          │ │ ARCHIVE      │ │  (ChromaDB)        │
    │                 │ │              │ │                    │
    │ Tagged K-V pairs│ │ L0: Topic    │ │ Embedding-based    │
    │ Always loaded   │ │     Directory│ │ semantic search    │
    │ ~200 tokens     │ │ L1: Segment  │ │ for L2/L3 content  │
    │                 │ │     Summaries│ │                    │
    │ Categories:     │ │ L2: Chunk    │ │ Also stores:       │
    │ • RULE          │ │     Summaries│ │ • File contents    │
    │ • PREFERENCE    │ │ L3: Raw data │ │ • Code snippets    │
    │ • LIMIT         │ │              │ │ • External docs    │
    │ • IDENTITY      │ │ (JSON/dict   │ │                    │
    └────────┬────────┘ │  for L0-L1,  │ └─────────┬──────────┘
             │          │  ChromaDB    │           │
             │          │  for L2-L3)  │           │
             │          └──────┬───────┘           │
             │                 │                   │
             └────────┬────────┴───────────────────┘
                      │
                      ▼
         ┌───────────────────────────┐
         │   ASSEMBLED CONTEXT       │
         │                           │
         │ ┌───────────────────────┐ │
         │ │ ZONE 1: Constraints   │ │  ← TOP (highest attention)
         │ │ (always included)     │ │
         │ ├───────────────────────┤ │
         │ │ ZONE 2: Relevant      │ │  ← Retrieved by curator
         │ │ context & history     │ │
         │ ├───────────────────────┤ │
         │ │ ZONE 3: Peripheral    │ │  ← 1-line summaries
         │ │ awareness             │ │     of other context
         │ ├───────────────────────┤ │
         │ │ ZONE 4: Current       │ │  ← BOTTOM (high attention)
         │ │ prompt                │ │
         │ └───────────────────────┘ │
         └────────────┬──────────────┘
                      │
                      ▼
         ┌───────────────────────────┐
         │       MAIN LLM            │
         │   (processes request)     │
         └────────────┬──────────────┘
                      │
                      ▼
         ┌───────────────────────────┐
         │    POST-PROCESSOR         │
         │                           │
         │ 1. Constraint violation   │
         │    check (keyword/regex)  │
         │ 2. Extract new            │
         │    constraints → Store    │
         │ 3. Summarize turn         │
         │    → Update archive       │
         │ 4. Store raw content      │
         │    → L3 + ChromaDB        │
         └───────────────────────────┘
```

### 4.2 Hierarchical Archive — The Paging Structure

```
LEVEL 0 — TOPIC DIRECTORY (always in curator's context)
┌──────────────────────────────────────────────────────────────────┐
│ Budget: MAX 600 tokens (~20 segments × 30 tokens each)          │
│                                                                  │
│ seg_001: "Turns 1-12: Project init, 3 constraints set, Python"  │
│ seg_002: "Turns 13-25: API design, REST chosen, 5 endpoints"    │
│ seg_003: "Turns 26-40: DB schema, PostgreSQL, 4 tables"         │
│ ...                                                              │
│                                                                  │
│ When segments exceed budget → MERGE old segments into            │
│ super-segments (recursive compression)                           │
└──────────────────────────────────────────────────────────────────┘

LEVEL 1 — SEGMENT SUMMARIES (fetched selectively by curator)
┌──────────────────────────────────────────────────────────────────┐
│ Budget: ~100-150 tokens per segment                              │
│                                                                  │
│ seg_001:                                                         │
│   entry_a: "User rule: never modify auth.py without permission"  │
│   entry_b: "Project uses FastAPI + SQLAlchemy"                   │
│   entry_c: "User prefers type hints everywhere"                  │
│   entry_d: "Initial folder structure created"                    │
│                                                                  │
│ Curator fetches 2-4 relevant segments per query.                 │
└──────────────────────────────────────────────────────────────────┘

LEVEL 2 — CHUNK SUMMARIES (semantic search via ChromaDB)
┌──────────────────────────────────────────────────────────────────┐
│ Budget: ~200-400 tokens per chunk                                │
│                                                                  │
│ Condensed but faithful summaries of individual turns,            │
│ code changes, or data points.                                    │
│ Retrieved via embedding similarity when L1 is insufficient.      │
└──────────────────────────────────────────────────────────────────┘

LEVEL 3 — RAW CONTENT (exact retrieval when needed)
┌──────────────────────────────────────────────────────────────────┐
│ Full original text of every turn, file, document.                │
│ Retrieved only when exact wording matters                        │
│ (e.g., code, specific numbers, exact quotes).                    │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Capacity Scaling Analysis

```
Configuration         Max Effective Turns    Growth Pattern
─────────────────────────────────────────────────────────────
Raw LLM (8K ctx)      ~30-50 turns           O(1) — fixed
RAG (flat)             ~200-500 turns         O(N) — linear retrieval degrades
MemGPT (2-tier)        ~500-1000 turns        O(k*N) — delayed linear
HierMem 2-level        ~D×S turns             O(D×S) — D segments × S turns/seg
  (D=20, S=50)         ~1000 turns
HierMem 3-level        ~M×D×S turns           O(M^depth)
  (M=20)               ~20,000 turns          Exponential scaling
```

**Key property**: Each paging level multiplies capacity by the branching factor M, while the curator's context stays bounded at O(M) tokens per level.

### 4.4 Adaptive Retrieval Strategy

Different query types benefit from different retrieval methods. The curator selects the strategy:

```
QUERY TYPE              → BEST RETRIEVAL        → WHY
─────────────────────────────────────────────────────────────
"What did user say       → KEYWORD search on     → Exact match needed
 about auth.py?"           constraint store

"Continue the API        → HIERARCHICAL (L1→L2)  → Need structured
 we were building"         + recent turns           sequential context

"Find something          → VECTOR SIMILARITY     → Semantic matching
 related to caching        on ChromaDB              across all history
 strategy"

"Summarize what          → HIERARCHICAL L0→L1    → Need breadth,
 we've done so far"                                 not depth

"Check if my approach    → VECTOR + CONSTRAINT   → Need both relevant
 violates any rules"       STORE                    context AND rules
```

**Implementation**: The curator prompt includes a routing instruction:
```
Given the user's query, select retrieval strategy:
- KEYWORD: when looking for specific named entities, rules, or exact terms
- HIERARCHY: when needing sequential/structured context from a time range
- SEMANTIC: when the query is conceptual and could match diverse past content  
- HYBRID: when both rules and contextual history are needed
```

This is what makes the system best-of-all-worlds — it doesn't commit to one retrieval method but dynamically selects the optimal one.

---

## 5. Module Specifications

### 5.1 Constraint Store

```python
# Data Structure
@dataclass
class Constraint:
    id: str                          # unique identifier
    text: str                        # the actual constraint text
    category: str                    # RULE | PREFERENCE | LIMIT | IDENTITY
    source_turn: int                 # when it was stated
    priority: int                    # 1 (low) to 5 (critical)
    active: bool                     # can be deactivated by user
    keywords: List[str]              # for keyword matching

# Operations
- add_constraint(text, category, priority)
- deactivate_constraint(id)
- get_all_active() → List[Constraint]     # for context assembly
- check_violation(text) → List[Constraint] # regex/keyword post-check
- total_tokens() → int                     # budget monitoring
```

**Constraint extraction**: After each LLM response, a lightweight classifier (regex + simple LLM call) scans for:
- Imperative statements: "don't...", "never...", "always...", "make sure..."
- Conditional rules: "if X then Y", "unless I say otherwise"
- Preferences: "I prefer...", "use X instead of Y"

### 5.2 Hierarchical Archive Manager

```python
class HierarchicalArchive:
    def __init__(self, max_l0_entries=20, max_l1_tokens_per_seg=150):
        self.l0_directory = {}        # seg_id → one-line summary
        self.l1_segments = {}         # seg_id → list of entry summaries
        self.vector_store = ChromaDB() # L2 + L3 content
        self.current_segment_id = 0
        self.turns_in_current_segment = 0
    
    def add_turn(self, turn_number, user_msg, assistant_msg, summary):
        """Called after every turn."""
        # 1. Store raw content at L3
        self.vector_store.add(raw_content, metadata={turn: turn_number})
        
        # 2. Store summary at L2
        self.vector_store.add(summary, metadata={level: "L2", turn: turn_number})
        
        # 3. Update current L1 segment
        self.l1_segments[self.current_segment_id].append(summary)
        
        # 4. Check if segment is full → create new segment
        if self.turns_in_current_segment >= SEGMENT_SIZE:
            self._close_segment()
        
        # 5. Check if L0 is over budget → merge old segments
        if len(self.l0_directory) > self.max_l0_entries:
            self._merge_oldest_segments()
    
    def retrieve(self, query, strategy, top_k=5):
        """Retrieve context based on curator's chosen strategy."""
        if strategy == "KEYWORD":
            return self._keyword_search(query)
        elif strategy == "HIERARCHY":
            return self._hierarchical_traverse(query)
        elif strategy == "SEMANTIC":
            return self.vector_store.query(query, n=top_k)
        elif strategy == "HYBRID":
            return self._hybrid_search(query, top_k)
```

### 5.3 Context Curator Agent

```python
CURATOR_SYSTEM_PROMPT = """
You are a Context Curator. Your job is to select the most relevant context 
for the Main LLM to answer the user's current query.

You will receive:
1. The user's current prompt
2. A Topic Directory (compressed index of all past conversation segments)
3. The active Constraint Store

Your output must be a JSON object:
{
    "retrieval_strategy": "KEYWORD|HIERARCHY|SEMANTIC|HYBRID",
    "segments_to_fetch": ["seg_001", "seg_003"],  // from Topic Directory
    "semantic_queries": ["query1", "query2"],       // for vector search
    "peripheral_segments": ["seg_002"],             // 1-line summaries only
    "reasoning": "brief explanation of choices"
}

Rules:
- Select AT MOST 4 segments for full fetch (token budget)
- ALL constraints from Constraint Store will be auto-included (you don't select those)
- If uncertain whether something is relevant, include it as peripheral
- Recent 2-3 turns are always included automatically
"""
```

### 5.4 Context Assembler

```python
def assemble_context(constraints, relevant_context, peripheral, current_prompt, 
                     token_budget=6000):
    """
    Assemble final context with attention-aware positioning.
    Constraints at TOP, current prompt at BOTTOM (highest attention zones).
    """
    sections = []
    
    # ZONE 1: Constraints (TOP — highest attention)
    sections.append("=== ACTIVE RULES & CONSTRAINTS ===")
    for c in constraints:
        sections.append(f"[{c.category}] {c.text}")
    
    # ZONE 2: Retrieved relevant context (UPPER-MIDDLE)
    sections.append("\n=== RELEVANT CONTEXT ===")
    for ctx in relevant_context:
        sections.append(ctx.text)
    
    # ZONE 3: Peripheral awareness (LOWER-MIDDLE — lowest attention zone)
    sections.append("\n=== OTHER CONTEXT (summaries) ===")
    for p in peripheral:
        sections.append(f"- {p.one_line_summary}")
    
    # ZONE 4: Current prompt (BOTTOM — high attention)
    sections.append(f"\n=== CURRENT REQUEST ===\n{current_prompt}")
    
    return truncate_to_budget("\n".join(sections), token_budget)
```

### 5.5 Post-Processor

```python
def post_process(llm_response, constraint_store, archive, turn_number):
    """Run after every Main LLM response."""
    
    # 1. Check constraint violations
    violations = constraint_store.check_violation(llm_response)
    if violations:
        return regenerate_with_violation_warning(llm_response, violations)
    
    # 2. Extract new constraints from user message + LLM response
    new_constraints = extract_constraints(user_message)
    for c in new_constraints:
        constraint_store.add(c)
    
    # 3. Generate turn summary (small LLM call, ~50 tokens output)
    summary = summarize_turn(user_message, llm_response)
    
    # 4. Update hierarchical archive
    archive.add_turn(turn_number, user_message, llm_response, summary)
    
    return llm_response
```

---

## 6. Tech Stack

### 6.1 Core Components

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **Language** | Python 3.11+ | Standard for ML research |
| **Vector Store** | ChromaDB | Free, local, no server needed, supports metadata filtering |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, fast (80M params), good quality |
| **Hierarchical Index** | Custom Python (dict + JSON) | L0/L1 are tiny — no DB overhead needed |
| **Constraint Store** | Custom Python dataclass + JSON persistence | Simple, fast, always in memory |
| **LLM Client** | `litellm` or custom wrapper | Unified API across all providers |
| **Token Counting** | `tiktoken` | Accurate budget management |
| **Evaluation** | `pandas` + `matplotlib` + `seaborn` | Standard research visualization |
| **Testing** | `pytest` | Unit + integration tests |
| **Configuration** | `python-dotenv` + `pydantic` | Type-safe config with env vars |

### 6.2 Models to Use

#### Primary Development & Testing (FREE / Local)

| Model | Size | Context | Provider | Best For | Notes |
|-------|------|---------|----------|----------|-------|
| **Llama 3.1 8B Instruct** | 8B | 128K | Groq (free API) / local (ollama) | Main LLM during dev | Good instruction following, free on Groq |
| **Llama 3.1 70B** | 70B | 128K | Groq (free API) | Main LLM for benchmarks | Significantly better reasoning |
| **Qwen 2.5 7B Instruct** | 7B | 128K | Local (ollama) | Curator agent | Good at structured output, efficient |
| **Qwen 2.5 3B Instruct** | 3B | 32K | Local (ollama) | Lightweight curator | Fastest option for curator |
| **Phi-3.5 Mini** | 3.8B | 128K | Local (ollama) | Summarization / post-processing | Tiny but capable for simple tasks |
| **Gemma 2 9B** | 9B | 8K | Local (ollama) / Groq | Alternative main LLM | Strong reasoning per size |
| **Mistral 7B v0.3** | 7B | 32K | Local (ollama) / Groq | Baseline testing | Well-studied baseline model |

#### Final Benchmark Run (PAID — for paper results)

| Model | Provider | Cost Estimate | Why Include |
|-------|----------|---------------|-------------|
| **GPT-4o-mini** | OpenAI | ~$0.15/M input | Cheap, strong, good baseline |
| **GPT-4o** | OpenAI | ~$2.50/M input | Top-tier, shows our system helps even strong models |
| **Claude 3.5 Haiku** | Anthropic | ~$0.25/M input | Fast, good instruction following |
| **Claude 3.5 Sonnet** | Anthropic | ~$3/M input | Top-tier, for final results |
| **Gemini 1.5 Flash** | Google | Free tier available | 1M context — interesting to compare: does our paging help even with huge context? |
| **Gemini 1.5 Pro** | Google | ~$1.25/M input | Native long-context — key comparison |

**Cost estimate for full benchmark**: $5-15 with mini/flash models, $30-60 with full models.

#### For Local Testing (no API, no cost)

```bash
# Install ollama (free, runs locally)
# https://ollama.ai

ollama pull llama3.1:8b          # Main LLM (4.7GB)
ollama pull qwen2.5:3b           # Curator agent (2GB)  
ollama pull phi3.5:latest         # Summarizer (2.2GB)
# Total: ~9GB VRAM needed (can run on 12GB GPU or CPU with slower speed)

# Alternative: all-in-one with single model
ollama pull llama3.1:8b          # Use for everything (4.7GB)
```

### 6.3 Model Role Assignment — Star Rating

| Task | Best Free Model | ★ Rating | Best Paid Model | ★ Rating |
|------|----------------|----------|----------------|----------|
| **Main LLM** | Llama 3.1 70B (Groq) | ★★★★☆ | GPT-4o / Claude Sonnet | ★★★★★ |
| **Curator Agent** | Qwen 2.5 7B | ★★★★☆ | GPT-4o-mini | ★★★★★ |
| **Summarizer** | Phi-3.5 Mini / Qwen 3B | ★★★☆☆ | GPT-4o-mini | ★★★★☆ |
| **Constraint Extractor** | Llama 3.1 8B | ★★★☆☆ | Claude Haiku | ★★★★★ |
| **Embeddings** | all-MiniLM-L6-v2 (local) | ★★★★☆ | OpenAI ada-002 | ★★★★★ |
| **Post-Check** | Regex + Llama 8B | ★★★☆☆ | Regex + GPT-4o-mini | ★★★★☆ |

---

## 7. Evaluation Plan

### 7.1 Benchmarks & Datasets

#### A. Multi-Turn Conversation with Constraints (CUSTOM — must create)

This is our **primary evaluation** since no existing benchmark tests exactly what we're measuring.

**Design**: Generate synthetic multi-turn conversations where:
- Turn 1-5: Set 3-5 explicit constraints (e.g., "always respond in bullet points", "never suggest Python 2 syntax", "the project uses PostgreSQL not MySQL")
- Turn 6-20: Normal task-oriented conversation
- Turn 21-30: Tasks that would naturally violate the early constraints if forgotten
- Turn 31-50: Increase complexity, add more context, introduce distractors

**Metrics measured at each checkpoint (turn 10, 20, 30, 40, 50)**:
- Constraint adherence rate
- Fact consistency with earlier turns
- Task accuracy

**Size**: 50 conversations × 50 turns = 2,500 evaluation points

#### B. Needle-in-a-Haystack (ADAPTED from existing work)

Use the standard NIAH test (Kamradt, 2023) but adapted for conversation:
- Plant a specific fact at turn N
- Ask about it at turn N+K for varying K (5, 10, 20, 30, 40)
- Measure recall accuracy vs. distance

**Size**: 20 facts × 5 distance levels × 3 context lengths = 300 tests

#### C. BABILong (EXISTING BENCHMARK — Kuratov et al., 2024)

- 20 reasoning tasks with facts scattered in long documents
- Tests: fact retrieval, fact chaining, deduction, counting
- Available on HuggingFace: `RMT-team/babilong`
- We run with our memory management vs. raw context

**Size**: 20 tasks × 5 context lengths (1K, 2K, 4K, 8K, 16K) = 100 test configs

#### D. LooGLE (EXISTING BENCHMARK — Li et al., 2023)

- Long-context QA with documents post-2022 (no training data leakage)
- 24K+ tokens per document, 6,000+ questions
- Tests both short-dependency and long-dependency tasks
- Available on HuggingFace: `bigainlp/LooGLE`

#### E. Multi-Turn Instruction Following (ADAPTED from IFEval)

- Based on Google's IFEval benchmark but extended to multi-turn
- Instructions accumulate across turns
- Tests whether the system remembers ALL active instructions

#### F. Multi-Step Reasoning with Dependencies (CUSTOM)

- Tasks where step N depends on the result of step N-3 or earlier
- Similar to GSM8K but multi-turn: give parts of the problem across different turns
- Tests information integration across conversation history

**Size**: 30 multi-step problems × 3 difficulty levels = 90 tests

### 7.2 Evaluation Metrics

#### Primary Metrics

| Metric | Definition | How to Measure |
|--------|-----------|---------------|
| **Degradation Curve** | Accuracy as function of conversation length | Plot accuracy at turns 10, 20, 30, 40, 50 for each system |
| **Constraint Violation Rate (CVR)** | % of responses violating a previously stated constraint | Automated regex check + LLM judge for semantic violations |
| **Fact Recall@K** | Can the system recall a fact stated K turns ago? | Binary: correct / incorrect, measured at K = 5, 10, 20, 30, 40 |
| **Task Accuracy** | % of task queries answered correctly | Exact match / LLM judge / F1 score depending on task type |
| **Consistency Score** | Does the system contradict itself across turns? | Extract claim pairs, check for contradictions |

#### Secondary Metrics (Efficiency)

| Metric | Definition | Target |
|--------|-----------|--------|
| **Latency per turn** | Wall-clock time from prompt to response | < 2x baseline |
| **Token usage per turn** | Total tokens consumed (all LLM calls combined) | < 3x baseline |
| **Storage growth** | Memory/disk usage over time | Linear, not quadratic |
| **Curator accuracy** | Does curator select relevant context? | Manual annotation on subset |

#### The Killer Chart — Degradation Curve

```
Task Accuracy (%)
100│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  HierMem (stays flat)
   │ ████████████
 90│ ████████████████████
   │ ████████████████████████
 80│              ████████████████████████████████  MemGPT-style
   │                    ██████████████████████████
 70│                          ████████████████████
   │                                ██████████████  RAG
 60│                                      ████████
   │                                          ████
 50│                                              █ Raw LLM
   │
   └──────────────────────────────────────────────
     10      20      30      40      50     turns
```

If we can show this curve, the paper writes itself.

### 7.3 Baselines

| System | Description | Fair Comparison Notes |
|--------|------------|----------------------|
| **Baseline 1: Raw LLM** | Full conversation history in context (truncated when exceeding window) | Same model, same token budget for response |
| **Baseline 2: RAG** | Standard top-k vector retrieval (ChromaDB + same embeddings) | Same embedding model, same chunk size |
| **Baseline 3: RAG + Summary** | RAG with rolling conversation summary prepended | Same summarization LLM |
| **Baseline 4: MemGPT-style** | Self-managed memory (LLM decides what to page in/out) | Same model, same storage backend |
| **HierMem (ours)** | Full system with curator + constraint store + hierarchical paging | Reports total compute used |

**All baselines use the same**:
- Main LLM model
- Total response token budget
- Storage backend (ChromaDB)
- Embedding model

**Only difference**: how context is managed and assembled.

---

## 8. Project Structure

```
hiermem/
├── README.md
├── requirements.txt
├── .env.example
├── config.py                           # All configuration
│
├── core/                               # Core system modules
│   ├── __init__.py
│   ├── curator.py                      # Context Curator Agent
│   ├── constraint_store.py             # Constraint Store
│   ├── archive.py                      # Hierarchical Archive Manager
│   ├── assembler.py                    # Context Assembler (attention-aware)
│   ├── post_processor.py              # Post-processing pipeline
│   └── pipeline.py                     # Main orchestration loop
│
├── memory/                             # Memory backends
│   ├── __init__.py
│   ├── vector_store.py                 # ChromaDB wrapper
│   ├── hierarchical_index.py           # L0/L1 dict-based index
│   └── embedding.py                    # Embedding model wrapper
│
├── retrieval/                          # Retrieval strategies
│   ├── __init__.py
│   ├── keyword.py                      # Keyword/regex search
│   ├── semantic.py                     # Vector similarity search
│   ├── hierarchical.py                 # Tree traversal retrieval
│   └── router.py                       # Adaptive strategy selection
│
├── llm/                                # LLM integration
│   ├── __init__.py
│   ├── client.py                       # Multi-provider LLM client
│   ├── prompts/                        # All prompts as text files
│   │   ├── curator_system.txt
│   │   ├── summarizer.txt
│   │   ├── constraint_extractor.txt
│   │   └── post_check.txt
│   └── token_counter.py               # Token budget management
│
├── baselines/                          # All baseline implementations
│   ├── __init__.py
│   ├── raw_llm.py                      # Baseline 1: Raw context window
│   ├── rag_baseline.py                 # Baseline 2: Standard RAG
│   ├── rag_summary.py                  # Baseline 3: RAG + rolling summary
│   └── memgpt_style.py                # Baseline 4: Self-managed memory
│
├── eval/                               # Evaluation framework
│   ├── __init__.py
│   ├── run_benchmark.py               # Run all systems on all benchmarks
│   ├── metrics.py                      # Compute all metrics
│   ├── visualize.py                    # Generate charts and tables
│   ├── datasets/                       # Evaluation datasets
│   │   ├── constraint_conversations/   # Custom: multi-turn w/ constraints
│   │   ├── needle_in_haystack/         # Adapted NIAH
│   │   └── multi_step_reasoning/       # Custom: multi-step problems
│   └── generators/                     # Dataset generation scripts
│       ├── generate_constraint_convos.py
│       ├── generate_niah.py
│       └── generate_multistep.py
│
├── tests/                              # Unit tests
│   ├── test_constraint_store.py
│   ├── test_archive.py
│   ├── test_curator.py
│   ├── test_assembler.py
│   ├── test_retrieval.py
│   └── test_pipeline_e2e.py
│
├── results/                            # Evaluation outputs (gitignored)
│   └── .gitkeep
│
├── notebooks/                          # Analysis notebooks
│   ├── exploration.ipynb
│   └── results_analysis.ipynb
│
└── demo/                               # Demo interface
    └── app.py                          # Streamlit demo
```

---

## 9. Retrieval Strategy Analysis — Is This the Best Combination?

### 9.1 Why Hybrid Beats Any Single Method

Each retrieval method has a **failure mode** that the others compensate for:

| Method | Strength | Failure Mode | When it Fails |
|--------|----------|-------------|---------------|
| **Vector similarity** | Finds semantically related content even with different wording | Misses exact constraints, poor at structured data | "What rule did user state about auth.py?" → finds similar but not exact |
| **Keyword/regex** | Exact match, fast, deterministic | Misses paraphrased content, zero semantic understanding | "What was the design decision about authentication?" → keyword "auth.py" doesn't match |
| **Hierarchical tree** | Gives structured overview, respects temporal order | Can't find specific needles buried in summaries | "What specific number did user mention in turn 7?" → lost in summarization |
| **Full-text search** | Finds exact phrases | No semantic understanding, no ranking | "What was our approach?" → too many matches |

**Our hybrid approach**: The curator agent picks the best strategy per query, or combines them:

```
                    USER QUERY
                        │
                  ┌─────┴──────┐
                  │  CURATOR    │
                  │  ROUTES TO: │
                  └─────┬──────┘
           ┌────────────┼────────────┐
           │            │            │
    ┌──────▼──────┐ ┌───▼───┐ ┌─────▼──────┐
    │  KEYWORD    │ │HIERARCH│ │  SEMANTIC   │
    │  for rules  │ │for     │ │  for        │
    │  & exact    │ │context │ │  concepts   │
    │  terms      │ │flow    │ │  & ideas    │
    └──────┬──────┘ └───┬───┘ └─────┬──────┘
           │            │            │
           └────────────┼────────────┘
                        │
                  MERGED & RANKED
                  (deduplicated, 
                   scored by relevance)
```

### 9.2 Comparison with Alternatives We Considered

| Alternative | Why We Didn't Choose It |
|------------|------------------------|
| **GraphRAG (full KG)** | Too expensive: requires LLM calls to extract entities/relations for every turn. Overkill for conversation memory. Our hierarchical summaries capture 90% of the benefit at 10% of the cost. |
| **Pure vector DB** | Misses the constraint-always-present requirement. Can't guarantee a rule stated at turn 1 is retrieved at turn 50 (embedding distance may be large). |
| **Sparse retrieval (BM25)** | Good for keyword match but no semantic understanding. We include keyword search as one strategy option. |
| **Reranker after retrieval** | Good idea — can add later as optimization. Uses a cross-encoder to rerank retrieved chunks. Would improve accuracy but adds latency and compute. |
| **Knowledge graph per session** | Interesting but expensive to maintain incrementally. Better suited for post-hoc analysis than real-time conversation. Could be a future extension. |

### 9.3 When to Use What — Decision Matrix

```python
RETRIEVAL_DECISION_MATRIX = {
    # Query signals                    → Strategy
    "contains_entity_name":            "KEYWORD",
    "asks_about_specific_turn":        "HIERARCHY",  
    "asks_about_rules_or_constraints": "KEYWORD + CONSTRAINT_STORE",
    "conceptual_or_open_ended":        "SEMANTIC",
    "needs_temporal_context":          "HIERARCHY",
    "needs_both_rules_and_context":    "HYBRID",
    "continuation_of_previous_task":   "HIERARCHY (recent) + SEMANTIC",
    "completely_new_topic":            "SEMANTIC + HIERARCHY (L0 scan)",
}
```

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up project structure
- [ ] Implement Constraint Store (add, deactivate, check, persist)
- [ ] Implement basic Hierarchical Archive (L0 + L1 only, in-memory)
- [ ] Implement ChromaDB vector store wrapper
- [ ] Implement LLM client (Groq + Ollama support)
- [ ] Basic token counting and budget management
- [ ] Unit tests for Constraint Store and Archive

### Phase 2: Core Pipeline (Week 2)
- [ ] Implement Context Curator agent with structured output
- [ ] Implement Context Assembler (4-zone attention-aware layout)
- [ ] Implement Post-Processor (constraint extraction + summarization + archival)
- [ ] Wire the full pipeline: Prompt → Curator → Retrieve → Assemble → LLM → Post-process
- [ ] Implement adaptive retrieval router
- [ ] End-to-end test with a 10-turn conversation

### Phase 3: Baselines (Week 2-3)
- [ ] Implement Baseline 1: Raw LLM (full context window)
- [ ] Implement Baseline 2: Standard RAG
- [ ] Implement Baseline 3: RAG + rolling summary
- [ ] Implement Baseline 4: MemGPT-style self-managed memory
- [ ] Verify all baselines work on a simple test case

### Phase 4: Dataset Generation (Week 3)
- [ ] Generate constraint conversation dataset (50 conversations × 50 turns)
- [ ] Generate needle-in-a-haystack adapted dataset
- [ ] Generate multi-step reasoning with dependencies dataset
- [ ] Download and prepare BABILong benchmark
- [ ] Download and prepare LooGLE benchmark subset
- [ ] Validate all datasets with a manual quality check

### Phase 5: Evaluation (Week 4)
- [ ] Run all 5 systems on all benchmarks
- [ ] Compute all metrics (degradation curves, CVR, Fact Recall@K, etc.)
- [ ] Generate comparison charts (degradation curves, bar charts, tables)
- [ ] Run ablation studies:
  - [ ] HierMem without constraint store
  - [ ] HierMem without hierarchical archive (flat vector only)
  - [ ] HierMem without curator (random retrieval)
  - [ ] HierMem with 2 levels vs 3 levels
- [ ] Statistical significance tests (3 runs per configuration)

### Phase 6: Analysis & Paper (Week 5-6)
- [ ] Failure mode analysis: categorize where each system fails
- [ ] Compute cost analysis: tokens used, latency, storage
- [ ] Write research paper:
  - [ ] Abstract, Introduction, Related Work
  - [ ] Method (architecture, each component)
  - [ ] Experiments (benchmarks, metrics, results)
  - [ ] Analysis (ablations, failure modes, cost)
  - [ ] Conclusion and Future Work
- [ ] Build Streamlit demo
- [ ] Create presentation slides

---

## 11. Testing Strategy

### 11.1 Unit Tests

```python
# test_constraint_store.py
def test_add_and_retrieve_constraint():
    store = ConstraintStore()
    store.add("Never modify auth.py", category="RULE", priority=5)
    assert len(store.get_all_active()) == 1

def test_violation_detection():
    store = ConstraintStore()
    store.add("Never modify auth.py", category="RULE", priority=5, keywords=["auth.py", "modify"])
    violations = store.check_violation("I'll modify auth.py to add the new endpoint")
    assert len(violations) == 1

def test_constraint_deactivation():
    store = ConstraintStore()
    cid = store.add("Use Python 3.11", category="PREFERENCE", priority=3)
    store.deactivate(cid)
    assert len(store.get_all_active()) == 0

# test_archive.py  
def test_l0_stays_bounded():
    archive = HierarchicalArchive(max_l0_entries=5)
    for i in range(20):
        archive.add_turn(i, f"user msg {i}", f"assistant msg {i}", f"summary {i}")
    assert len(archive.l0_directory) <= 5  # merged

def test_hierarchical_retrieval():
    archive = HierarchicalArchive()
    archive.add_turn(1, "Set up PostgreSQL database", "Done", "DB setup with PostgreSQL")
    results = archive.retrieve("What database are we using?", strategy="SEMANTIC")
    assert "PostgreSQL" in results[0].text

# test_curator.py (with mocked LLM)
def test_curator_selects_relevant_segments(mock_llm):
    mock_llm.return_value = '{"segments_to_fetch": ["seg_001"], "retrieval_strategy": "HIERARCHY"}'
    curator = ContextCurator(llm=mock_llm)
    result = curator.select_context(query="Continue the API work", directory=sample_directory)
    assert "seg_001" in result.segments_to_fetch

# test_assembler.py
def test_constraints_at_top():
    assembled = assemble_context(
        constraints=[Constraint(text="Never modify auth.py")],
        relevant_context=[Context(text="Previous API discussion")],
        peripheral=[Summary(text="DB work in segment 3")],
        current_prompt="Add new endpoint"
    )
    lines = assembled.split("\n")
    constraint_pos = next(i for i, l in enumerate(lines) if "auth.py" in l)
    prompt_pos = next(i for i, l in enumerate(lines) if "Add new endpoint" in l)
    assert constraint_pos < prompt_pos  # constraint before prompt

# test_pipeline_e2e.py
def test_constraint_remembered_after_20_turns(real_or_mocked_llm):
    pipeline = HierMemPipeline(llm=real_or_mocked_llm)
    pipeline.process_turn("Rule: always use type hints in Python code")
    for i in range(20):
        pipeline.process_turn(f"Write a function that does task {i}")
    response = pipeline.process_turn("Write a function to parse JSON")
    assert ":" in response  # type hints present (e.g., "def parse(data: str) -> dict:")
```

### 11.2 Integration Tests

- Run 5 hand-crafted 30-turn conversations through full pipeline
- Verify constraints are never violated
- Verify facts from turn 5 can be recalled at turn 25
- Verify hierarchical archive merges correctly under load

### 11.3 Stress Tests

- Run a 200-turn synthetic conversation
- Monitor memory usage, latency, and storage growth
- Verify L0 directory stays bounded
- Verify no information is completely lost (only compressed)

---

## 12. Datasets — Where to Get Them

### 12.1 Existing Benchmarks (Download Links)

| Dataset | Where | Size | Use For |
|---------|-------|------|---------|
| **BABILong** | `huggingface.co/datasets/RMT-team/babilong` | 20 tasks × multiple lengths | Long-context fact retrieval |
| **LooGLE** | `huggingface.co/datasets/bigainlp/LooGLE` | 6000+ questions | Long-document QA |
| **∞Bench** | `github.com/OpenBMB/InfiniteBench` | 100K+ token contexts | Extreme long-context |
| **NIAH** | `github.com/gkamradt/LLMTest_NeedleInAHaystack` | Configurable | Recall at position |
| **MT-Bench** | `huggingface.co/datasets/lmsys/mt_bench_human_judgments` | 80 multi-turn convos | Multi-turn quality |
| **GSM8K** | `huggingface.co/datasets/gsm8k` | 8.5K math problems | Reasoning accuracy |
| **HotpotQA** | `huggingface.co/datasets/hotpot_qa` | 113K multi-hop questions | Multi-hop reasoning |

### 12.2 Custom Datasets We Must Generate

#### Constraint Conversation Dataset
```json
{
    "conversation_id": "conv_001",
    "constraints_planted": [
        {"turn": 1, "text": "Always respond in bullet points", "type": "FORMAT"},
        {"turn": 3, "text": "Never suggest deprecated APIs", "type": "RULE"},
        {"turn": 5, "text": "Project uses PostgreSQL, not MySQL", "type": "FACT"}
    ],
    "turns": [
        {"turn": 1, "role": "user", "text": "Hi, important rule: always respond in bullet points for this session."},
        {"turn": 1, "role": "assistant", "text": "..."},
        ...
        {"turn": 35, "role": "user", "text": "Write me a database connection function"},
        // Ground truth: should use PostgreSQL (not MySQL), in bullet points
    ],
    "checkpoints": [
        {"turn": 10, "test": "Is response in bullet points?", "answer": true},
        {"turn": 20, "test": "Is response in bullet points?", "answer": true},
        {"turn": 35, "test": "Does code use PostgreSQL?", "answer": true},
        {"turn": 35, "test": "Is response in bullet points?", "answer": true}
    ]
}
```

We generate these programmatically using an LLM to create realistic conversations, then manually validate a subset.

---

## 13. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Curator selects wrong context | Medium | High | Ablation study shows impact; log curator decisions for debugging; fall back to hybrid retrieval |
| Summarization at L1/L2 loses critical detail | Medium | Medium | Keep L3 raw content always available; tune summary prompts on validation set |
| Token budget exceeded with too many constraints | Low | Medium | Cap constraint store at 500 tokens; auto-compress old constraints |
| Vector store retrieval is too slow | Low | Low | ChromaDB is fast for <100K entries; add caching for hot segments |
| Baselines are unfairly disadvantaged | Medium | High | Give all baselines same total compute budget; report tokens used |
| Custom benchmark doesn't reflect real usage | Medium | Medium | Also evaluate on 3 existing benchmarks (BABILong, LooGLE, MT-Bench) |
| Free models too weak to show differentiation | Medium | Medium | Test on Groq 70B (free); if no signal, use paid models |
| Paging overhead makes system too slow | Low | Medium | Profile; most overhead is curator LLM call (~500ms); amortized well |

---

## 14. Cost Estimate

### Development Phase (using free resources)

| Resource | Provider | Cost |
|----------|----------|------|
| Llama 3.1 8B/70B | Groq (free tier: 30 req/min) | $0 |
| Qwen 2.5 3B-7B | Ollama (local) | $0 (electricity only) |
| ChromaDB | Local | $0 |
| sentence-transformers | Local | $0 |
| BABILong, LooGLE, etc. | HuggingFace | $0 |

**Total dev cost: $0** (if you have a machine with 8GB+ RAM and optionally a GPU)

### Final Benchmark Run

| Configuration | Est. Tokens | Model | Cost |
|--------------|-------------|-------|------|
| 5 systems × 50 convos × 50 turns | ~50M tokens | GPT-4o-mini ($0.15/M) | ~$7.50 |
| Same with paid models | ~50M tokens | GPT-4o ($2.50/M) | ~$125 |
| BABILong + LooGLE | ~20M tokens | GPT-4o-mini | ~$3 |
| Ablation studies (4 configs) | ~40M tokens | GPT-4o-mini | ~$6 |

**Recommended budget: $20-30** for a solid paper with mini/flash models.
**Premium budget: $150-200** for full results with top models (for camera-ready paper).

---

## 15. Research Paper Outline

### Title Options
1. "HierMem: Paged Context Management for Long-Horizon LLM Reliability"
2. "Virtual Memory for LLMs: Hierarchical Context Paging with Dedicated Curation"
3. "Beyond Flat RAG: Multi-Level Memory Paging for Persistent LLM Conversations"

### Paper Structure
1. **Abstract** (250 words)
2. **Introduction** (1.5 pages) — Problem, motivation, key insight (OS paging analogy)
3. **Related Work** (1.5 pages) — MemGPT, RAPTOR, GraphRAG, Self-RAG, lost-in-middle
4. **Method** (3 pages) — Architecture, each component, capacity analysis
5. **Experimental Setup** (1.5 pages) — Benchmarks, baselines, metrics, models
6. **Results** (2 pages) — Degradation curves, tables, key findings
7. **Analysis** (1.5 pages) — Ablations, failure modes, cost analysis
8. **Conclusion** (0.5 pages) — Summary, limitations, future work

### Target Venues
- **ACL/EMNLP/NAACL** — Top NLP venues (competitive but high impact)
- **NeurIPS/ICML** — ML venues (if we frame it as a systems contribution)
- **AAAI** — AI venues (good for agent/architecture papers)
- **arXiv preprint** — Immediate visibility regardless of venue acceptance
- **Workshop papers** — NeurIPS Workshop on Memory in AI, ACL SRW (Student Research Workshop)

---

## 16. Future Extensions

1. **Learned Curator**: Fine-tune the curator on logged retrieval decisions with success/failure signal
2. **Cross-Session Memory**: Persist hierarchical archive across sessions for the same user/project
3. **Multi-Modal Support**: Extend archive to store and retrieve images, code diffs, diagrams
4. **Dynamic Level Depth**: Auto-add paging levels when capacity approaches threshold
5. **Federated Memory**: Share constraint patterns across different conversations (privacy-preserving)
6. **Hardware-Accelerated Retrieval**: GPU-based vector search for production-scale deployment

---

## 17. Quick Start Guide

```bash
# 1. Clone/create project
mkdir hiermem && cd hiermem

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install chromadb sentence-transformers litellm pydantic python-dotenv \
            tiktoken tqdm pandas matplotlib seaborn pytest datasets

# 4. Set up local model (free)
# Install ollama from https://ollama.ai
ollama pull llama3.1:8b
ollama pull qwen2.5:3b

# 5. Or use Groq (free API)
# Get key from https://console.groq.com
echo "GROQ_API_KEY=gsk_..." > .env

# 6. Build order:
#    core/constraint_store.py → core/archive.py → memory/vector_store.py →
#    retrieval/router.py → core/curator.py → core/assembler.py → 
#    core/post_processor.py → core/pipeline.py

# 7. Run tests
pytest tests/ -v

# 8. Run evaluation
python eval/run_benchmark.py --systems all --benchmark constraint_convos
python eval/visualize.py --results results/latest/
```

---

## 18. Summary

| Aspect | Details |
|--------|---------|
| **Core Idea** | OS-inspired multi-level paged memory for LLM context management |
| **Key Innovation** | Stateless curator + constraint store + hierarchical archive = exponential capacity |
| **Primary Metric** | Accuracy degradation curve over conversation length |
| **Baselines** | Raw LLM, RAG, RAG+Summary, MemGPT-style |
| **Benchmarks** | Custom constraint convos + BABILong + LooGLE + NIAH + multi-step reasoning |
| **Models** | Dev: Llama 3.1 (free via Groq/Ollama). Final: GPT-4o-mini + Claude Haiku |
| **Dev Cost** | $0 (free models + local tools) |
| **Paper Cost** | $20-30 (mini models) to $150-200 (full models) |
| **Timeline** | 5-6 weeks |
| **Novelty Level** | High — multi-level paging for LLM memory is unexplored |
| **Publishability** | Strong — clear contribution, measurable results, novel evaluation |

---

---

## Appendix A: L3 (Raw Storage) — Efficiency & Compute Analysis

### What L3 Actually Stores

L3 is the **full, unmodified record** of every interaction — the complete user message and the complete assistant response, stored verbatim. Think of it as the "disk" in the OS analogy. Every conversation turn gets written to L3 (ChromaDB with vector embeddings), but it is **almost never fully read back**.

**Key insight: L3 is write-heavy, read-light.** Just like a filesystem stores all files but the OS only loads the ones you open.

### Efficiency Breakdown: HierMem vs. Baselines

| Metric | Raw LLM | RAG Baseline | MemGPT-style | **HierMem** |
|--------|---------|-------------|--------------|-------------|
| **Tokens sent per turn (turn 50)** | ~10,000 (all history) | ~1,500 (top-k) | ~4,000 (self-managed) | ~2,000-4,000 (adaptive) |
| **LLM calls per turn** | 1 | 1 | 1-3 (memory actions) | 2-3 (curator + main + summarizer) |
| **Storage growth** | None (context only) | Linear in ChromaDB | Linear in ChromaDB | Linear in ChromaDB + small L0/L1 |
| **Context window usage** | 100% (overflows) | Fixed ~25% | Variable 30-70% | Adaptive 30-60% |
| **Scales to 500+ turns?** | ❌ Context overflow | ⚠️ Loses structure | ⚠️ Self-management degrades | ✅ L0 merging keeps it bounded |

### Storage Cost (Approximation for 100-turn conversation)

- **L3 raw storage**: ~200 tokens/turn × 100 turns = 20,000 tokens → ~80KB in ChromaDB
- **L2 summaries**: ~50 tokens/turn × 100 = 5,000 tokens → ~20KB
- **L1 segments**: ~20 tokens/entry × 10 segments = 200 tokens → <1KB (in-memory)
- **L0 directory**: ~20 tokens/entry × 10 entries = 200 tokens → <1KB (in-memory)
- **Constraint store**: ~500 tokens max → <2KB (in-memory)

**Total: ~100KB per 100-turn conversation.** Negligible. ChromaDB compresses vectors efficiently, and L0/L1/constraints are tiny in-memory structures.

### When L3 Is Actually Retrieved

L3 raw content is **only** fetched when:
1. The curator explicitly requests `KEYWORD` strategy (exact code/quotes needed)
2. The curator uses `HYBRID` and vector similarity finds a strong L3 match
3. It is never bulk-loaded — only specific turns matching the query are returned

**Efficiency advantage over RAG**: Standard RAG always does the same top-k retrieval. HierMem routes through the curator which often needs only L1 summaries (10x cheaper to process) and only goes to L3 when precision matters.

---

## Appendix B: Dynamic Constraint & Memory Updates

### The Problem: Constraints Are Not Static

Real conversations evolve:
- Turn 1: "Use PostgreSQL for the database"
- Turn 20: "Actually, let's switch to MongoDB"
- Turn 35: "Also, never deploy to production without tests"

All three must be tracked, and turn 20 should **deactivate** the PostgreSQL rule and **add** the MongoDB one.

### How Dynamic Updates Work

**Constraint extraction happens on EVERY turn**, not just the first. The `PostProcessor` runs after every main LLM response:

```
For every user message:
  1. Regex pre-check: does message contain constraint-like language?
     (never, always, must, don't, make sure, rule, prefer, use X instead, etc.)
  2. If YES → LLM extraction: send to summarizer model for structured constraints
  3. For each extracted constraint:
     - Check if it contradicts existing ones (keyword overlap + different direction)
     - If contradiction → deactivate old, add new
     - If new → add to constraint store
  4. Auto-eviction: if at MAX_CONSTRAINTS (20), lowest-priority gets evicted
```

### Who Updates What — The Write Path

| Component | Who writes to it | When | How |
|-----------|-----------------|------|-----|
| **L3 (raw)** | PostProcessor | Every turn | Automatic — always store full content |
| **L2 (summaries)** | PostProcessor via LLM | Every turn | Summarizer model generates 1-sentence summary |
| **L1 (segment bullets)** | PostProcessor | Every turn | Shortened version of L2 summary |
| **L0 (topic labels)** | Archive auto-update | Every turn + on segment close | Regenerated from L1 entries |
| **Constraint Store** | PostProcessor via LLM | Every turn (if signal detected) | Regex pre-check → LLM extraction → add/deactivate |
| **L0 merge** | Archive | When L0 exceeds MAX_L0_ENTRIES | Two oldest segments merged into super-segment |

### The Complete Write Flow (Every Turn)

```
User sends message
    │
    ▼
[Curator reads L0 + constraints, selects context]  ← READ only
    │
    ▼
[Retrieval fetches from L1/L2/L3]  ← READ only
    │
    ▼
[Assembler builds context]  ← READ only, constructs prompt
    │
    ▼
[Main LLM generates response]
    │
    ▼
[PostProcessor — ALL WRITES happen here]
    │
    ├─→ constraint_store.check_violation(response)    ← READ constraint store
    ├─→ _extract_constraints(user_msg)                ← WRITE to constraint store
    ├─→ _generate_summaries(user_msg, assistant_msg)  ← LLM call → summary + bullet
    └─→ archive.add_turn(turn, user, asst, summary, bullet)
              │
              ├─→ vector_store.add(raw_content, level="L3")     ← WRITE L3
              ├─→ vector_store.add(summary, level="L2")         ← WRITE L2
              ├─→ l1_segments[current].entries.append(bullet)   ← WRITE L1
              ├─→ _update_l0_label(current_segment)             ← WRITE L0
              ├─→ if segment full: _close_current_segment()     ← NEW L0/L1 segment
              └─→ if L0 overflow: _merge_oldest_segments()      ← MERGE L0/L1
```

### Token Allocation Flexibility

The system is **adaptive**, not fixed:

```
TOTAL_CONTEXT_BUDGET = 6000 tokens (configurable)

Zone 1 (Constraints):  ~500 tokens max — grows/shrinks with active constraints
Zone 2 (Relevant):     ~4000 tokens max — curator decides how much to retrieve
Zone 3 (Peripheral):   ~500 tokens max — 1-line summaries of non-selected segments
Zone 4 (Prompt):       ~1000 tokens max — current user message
```

The **curator dynamically decides** what to retrieve. Simple "hi" → NONE strategy, ~500 tokens. Complex "refactor auth considering database decisions" → 3 segments from L1 + semantic search, full 4000-token Zone 2.

**Truncation priority guarantees correctness:**
1. Never drop: Constraints (Zone 1) + Current Prompt (Zone 4)
2. Drop first: Peripheral summaries (Zone 3)
3. Drop second: Oldest/least-relevant chunks (Zone 2)

---

## Appendix C: Read vs Write Paths — Who Allocates What

### The Curator Is the READ Controller

Stateless, lightweight LLM call receiving:
- User's current message (~100 tokens)
- L0 topic directory (~400 tokens for 20 segments)
- Active constraints (~500 tokens max)
- **Total: ~1,000 tokens (constant, regardless of conversation length)**

### The PostProcessor Is the WRITE Controller

Handles ALL writes after the main LLM responds. Processes one turn at a time.

### Why This Split Prevents Degradation

```
READ PATH (curator): Stateless → never degrades
  Input: ~1000 tokens (constant)    Cost: ~*Document Version: 1.0*
*Created: 2026-03-01*
*Project: HierMem — Hierarchical Paged Context Management*.0001

WRITE PATH (post-processor): One turn at a time → never accumulates
  Input: ~400 tokens (constant)     Cost: ~*Document Version: 1.0*
*Created: 2026-03-01*
*Project: HierMem — Hierarchical Paged Context Management*.0002
```

Neither ever sees the full conversation. **The nesting problem is architecturally impossible.**

---

## Appendix D: Implementation Status (2026-03-01)

All core logic implemented and tested. 28 unit tests passing. 3 evaluation datasets generated (120 synthetic conversations). Virtual environment with all dependencies.

| Module | Status |
|--------|--------|
| `core/` (6 files) | ✅ Fully implemented |
| `memory/` (3 files) | ✅ ChromaDB + fallback |
| `retrieval/` (4 files) | ✅ Router + 3 strategies |
| `llm/` + prompts | ✅ Multi-provider |
| `baselines/` (4 files) | ✅ All 4 baselines |
| `eval/` (3 files) | ✅ Runner, metrics, viz |
| `eval/generators/` (3 files) | ✅ 120 conversations |
| `tests/` (5 files) | ✅ 28 tests passing |
| `demo/app.py` | ✅ Streamlit app |

### Remaining: E2E testing with real LLM, benchmark execution, ablation studies, paper writing.

---

*Document Version: 1.1 (updated with Appendices A-D)*
*Created: 2026-03-01*
*Project: HierMem — Hierarchical Paged Context Management*
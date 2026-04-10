# Gemini Nano Banana Pro Prompt (Camera-Ready v2): HierMem Architecture Figure

Use this exact prompt with Gemini Nano Banana Pro to generate a publication-ready architecture diagram.

## Copy-Paste Prompt

Design a research-paper architecture diagram for:
"HierMem: Constraint-Preserving Hierarchical Context Management for Long-Horizon LLM Conversations"

I need a clean, high-detail, conference-quality figure (ACL/EMNLP/NeurIPS style), not a marketing infographic.

Hard requirements:
1. White background, crisp vector-like look, no glossy effects, no 3D, no decorative icons.
2. Typography must be compact and legible at single-column paper scale.
3. Clear left-to-right pipeline with minimal edge crossings.
4. Consistent stroke widths and consistent arrowheads.
5. Include explicit stage headers for all stages 1..8 (no missing stage label).
6. No typos anywhere in labels, arrows, or callouts.

Strict color semantics (do not deviate):
- Constraint/invariant path and constraint-only boxes: deep red (#B42318)
- Curator/control path: deep blue (#1D4ED8)
- Retrieval/memory path: deep green (#15803D)
- Main generation path: charcoal (#374151)
- Neutral containers and generic non-constraint inputs: slate gray (#94A3B8)

Do NOT color non-constraint inputs as constraint red.

System content to render exactly:

Stage 1: Inputs
- User Turn t
- Prior Conversation State
- Active Constraint Store

Stage 2: Curator Orchestrator (small model)
- Box title: "Curator LLM (3B)"
- Inputs: current prompt, L0 topic directory, active constraints summary
- Outputs fields:
  - retrieval_strategy = {recent, targeted, semantic, hybrid}
  - segments_to_fetch
  - semantic_queries
  - peripheral_segments
  - fetch_full_turns

Stage 3: Hierarchical Memory Plane (stacked)
- L0 Topic Directory (segment index)
- L1 Segment Summaries
- L2 Embedding Index (vector store)
- L3 Raw Turns (verbatim archive)

Routing arrows (explicit labels):
- targeted -> L1/L3
- semantic -> L2 then L3
- hybrid -> L1 + L2 + L3

Stage 4: Adaptive Gate
- Decision diamond with TWO explicit outputs and labels:
  - path A: if history_tokens < passthrough_threshold -> Full-History Passthrough
  - path B: else -> Curated Hierarchical Retrieval
- Both paths must visibly rejoin into Stage 5 Context Assembler.

Stage 5: Context Assembler (bounded budget)
- Show stage header text explicitly: "Stage 5: Context Assembler"
- Box title: "Context Assembler"
- Render 4 prompt zones in strict order:
  1) Invariant Constraints (protected, non-evictable)
  2) Curated Relevant Context
  3) Peripheral Summaries
  4) Current User Request

Stage 6: Main Inference
- Box title: "Main LLM (benchmark model, e.g. Qwen2.5-14B)"
- Input: assembled prompt
- Output: assistant response

Stage 7: Post-Processor
- Constraint extraction from latest user turn
- Violation check on assistant response
- Optional retry path if violation detected

Stage 8: Feedback + Archive Update
- Update L0/L1/L2/L3
- Close loop back to Curator and Memory Plane

Callout panels (exactly two, no duplicates):

Panel A: Metrics Snapshot
- Mean judge score: 8.461 (best)
- Mean compute cost/turn: 0.0176 (lowest)
- Constraint survival: 93.3%
- Benchmarks: 15 conversations, 50 turns each

Panel B: Design Thesis
- "Constraint visibility is structurally guaranteed."
- "Main-model context growth is bounded via hierarchical curation."

Hard layout constraints:
- Keep Stage 3 memory plane visually central and prominent.
- Keep Stage 2 curator and Stage 5 assembler visually distinct.
- Keep Stage 4 gate centered between Stage 3 and Stage 5.
- Avoid overlap and arrow crossings where possible.
- Keep label density manageable: shorten any overlong text while preserving meaning.
- Ensure the feedback loop label is clean and typo-free.

Quality checks before final output:
- No duplicated panel A or panel B.
- No misspelled arrow labels.
- Stage 1..8 all visible.
- Stage 5 header explicitly present.

Output variants:
1) Wide landscape version for main paper figure.
2) Compact vertical version for single-column placement.

Do not omit any stage or any L0/L1/L2/L3 level.

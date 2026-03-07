"""
Cost & Compute Analysis — derives resource usage from existing results.json.

Metrics computed per system:
  - LLM calls per turn  (curator + main for hiermem, 1 for raw_llm/rag, 2 for rag_summary)
  - Input context tokens per turn
  - Wall-clock latency per turn
  - Vector DB queries per turn  (from sources_used entries)
  - Estimated API cost at GPT-4o-mini / Claude Haiku rates

Storage overhead (estimated):
  - raw_llm:    none
  - rag:        ChromaDB embeddings (N turns × 1536 dims × 4 bytes)
  - rag_summary: ChromaDB + rolling summary text
  - hiermem:    ChromaDB (L2/L3) + L0/L1 directory + constraint store

Usage:
  python -m eval.cost_analysis results/raw/benchmarks/qwen14b_run_v2
  python -m eval.cost_analysis results/raw/benchmarks/qwen14b_run_v2 --costs
"""

import json
import sys
import argparse
from pathlib import Path
from statistics import mean, median


# ----- LLM call counts per system per turn ------
# These are fixed structural costs, not measured (because Ollama doesn't report them).
# hiermem:     1 curator call (when HYBRID, not passthrough) + 1 main call
# rag_summary: 1 main call + 1 summary-update call
# rag / raw_llm: 1 main call
EXTRA_CALLS = {
    "hiermem": "variable",   # 1 passthrough, 2 when HYBRID
    "raw_llm": 1,
    "rag": 1,
    "rag_summary": 2,
    "memgpt_style": 2,
}

# API pricing (USD per 1M tokens) for cost projection
PRICING = {
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4o":           {"input": 2.50,  "output": 10.00},
    "claude-haiku-3.5": {"input": 0.80,  "output": 4.00},
    "gemini-flash":     {"input": 0.075, "output": 0.30},
}

# Rough output token estimate (responses aren't logged by token, use char/4 heuristic)
CHARS_PER_TOKEN = 4


def count_vector_queries(sources: list) -> int:
    """Count ChromaDB/vector queries from the sources_used list."""
    return sum(1 for s in sources if "semantic" in str(s) or "L2:" in str(s) or "L3:" in str(s))


def analyze_system(system_name: str, results: list) -> dict:
    """Analyze one system's results across all conversations."""
    all_turns = []
    for conv in results:
        turn_logs = conv.get("turn_logs", [])
        for turn in turn_logs:
            if "error" in turn and "ERROR:" in str(turn.get("response", "")):
                continue  # skip errored turns

            entry = {
                "turn": turn.get("turn", 0),
                "latency_s": turn.get("latency_seconds", 0),
                "context_tokens": 0,
                "llm_calls": EXTRA_CALLS.get(system_name, 1),
                "vector_queries": 0,
                "response_chars": len(turn.get("response", "")),
                "curator_mode": None,
            }

            # Context tokens
            pd = turn.get("pipeline_details", {})
            if pd:
                entry["context_tokens"] = pd.get("context_tokens_used", 0)
                sources = pd.get("sources_used", [])
                entry["vector_queries"] = count_vector_queries(sources)
                strategy = pd.get("curator_strategy")
                entry["curator_mode"] = strategy
                # HierMem: passthrough = 1 LLM call, HYBRID = 2 LLM calls
                if system_name == "hiermem":
                    warnings = pd.get("warnings", [])
                    is_passthrough = any("passthrough" in w for w in warnings)
                    entry["llm_calls"] = 1 if is_passthrough else 2
            elif "context_tokens" in turn:
                entry["context_tokens"] = turn["context_tokens"]

            all_turns.append(entry)

    if not all_turns:
        return {}

    total_turns = len(all_turns)
    total_llm_calls = sum(t["llm_calls"] for t in all_turns)
    total_context_tokens = sum(t["context_tokens"] for t in all_turns)
    total_latency = sum(t["latency_s"] for t in all_turns)
    total_vector_queries = sum(t["vector_queries"] for t in all_turns)
    total_output_chars = sum(t["response_chars"] for t in all_turns)
    estimated_output_tokens = total_output_chars // CHARS_PER_TOKEN

    # Per-turn breakdowns
    latencies = [t["latency_s"] for t in all_turns]
    ctx_tokens = [t["context_tokens"] for t in all_turns]

    # LLM call breakdown for hiermem
    passthrough_turns = sum(1 for t in all_turns if t["curator_mode"] is None and system_name == "hiermem")
    hybrid_turns = sum(1 for t in all_turns if t["curator_mode"] == "HYBRID")

    return {
        "total_turns": total_turns,
        "total_llm_calls": total_llm_calls,
        "avg_llm_calls_per_turn": round(total_llm_calls / total_turns, 2),
        "total_context_tokens": total_context_tokens,
        "avg_context_tokens_per_turn": round(total_context_tokens / total_turns),
        "avg_latency_s": round(mean(latencies), 1),
        "median_latency_s": round(median(latencies), 1),
        "total_latency_s": round(total_latency, 1),
        "avg_vector_queries_per_turn": round(mean(t["vector_queries"] for t in all_turns), 1),
        "estimated_output_tokens": estimated_output_tokens,
        # For cost calc: total input tokens ≈ context_tokens × llm_calls
        # (crude: curator call uses ~500 tokens, main call uses context_tokens)
        "total_input_tokens_approx": total_context_tokens + (
            500 * sum(1 for t in all_turns if t["llm_calls"] == 2)  # curator overhead
        ),
        "passthrough_turns": passthrough_turns if system_name == "hiermem" else None,
        "hybrid_turns": hybrid_turns if system_name == "hiermem" else None,
    }


def print_comparison_table(analyses: dict):
    systems = list(analyses.keys())
    if not systems:
        print("No systems found.")
        return

    col_w = max(18, max(len(s) for s in systems) + 2)

    def row(label, *vals):
        print(f"  {label:<35}" + "".join(str(v).rjust(col_w) for v in vals))

    print("\n" + "=" * (37 + col_w * len(systems)))
    print("  COMPUTE COST COMPARISON")
    print("=" * (37 + col_w * len(systems)))
    print(f"  {'Metric':<35}" + "".join(s.rjust(col_w) for s in systems))
    print("-" * (37 + col_w * len(systems)))

    def get(s, k, fmt=None):
        v = analyses.get(s, {}).get(k, "N/A")
        if fmt and v != "N/A":
            return fmt.format(v)
        return v

    row("Total turns processed",
        *[get(s, "total_turns") for s in systems])
    row("Total LLM calls",
        *[get(s, "total_llm_calls") for s in systems])
    row("Avg LLM calls / turn",
        *[get(s, "avg_llm_calls_per_turn", "{:.2f}") for s in systems])
    row("Avg context tokens / turn",
        *[get(s, "avg_context_tokens_per_turn") for s in systems])
    row("Total context tokens (input)",
        *[get(s, "total_context_tokens") for s in systems])
    row("Est. output tokens (chars/4)",
        *[get(s, "estimated_output_tokens") for s in systems])
    row("Avg vector DB queries / turn",
        *[get(s, "avg_vector_queries_per_turn", "{:.1f}") for s in systems])
    row("Avg latency / turn (sec)",
        *[get(s, "avg_latency_s", "{:.1f}s") for s in systems])
    row("Median latency / turn (sec)",
        *[get(s, "median_latency_s", "{:.1f}s") for s in systems])
    row("Total wall-clock time (min)",
        *["{:.1f}m".format(analyses.get(s, {}).get("total_latency_s", 0) / 60) for s in systems])

    # HierMem breakdown
    for s in systems:
        if s == "hiermem" and analyses.get(s, {}).get("hybrid_turns") is not None:
            a = analyses[s]
            print(f"\n  hiermem mode breakdown:")
            print(f"    Passthrough turns (1 LLM call): {a['passthrough_turns']}")
            print(f"    HYBRID turns (2 LLM calls):     {a['hybrid_turns']}")


def print_cost_projection(analyses: dict):
    systems = list(analyses.keys())
    col_w = max(18, max(len(s) for s in systems) + 2)

    print("\n" + "=" * (37 + col_w * len(systems)))
    print("  API COST PROJECTION (if using cloud models)")
    print("  Note: local Ollama = $0, this is for GPT/Claude comparison")
    print("=" * (37 + col_w * len(systems)))
    print(f"  {'Model':<35}" + "".join(s.rjust(col_w) for s in systems))
    print("-" * (37 + col_w * len(systems)))

    for model, prices in PRICING.items():
        costs = []
        for s in systems:
            a = analyses.get(s, {})
            if not a:
                costs.append("N/A")
                continue
            input_tok = a.get("total_input_tokens_approx", 0)
            output_tok = a.get("estimated_output_tokens", 0)
            cost = (input_tok / 1_000_000 * prices["input"] +
                    output_tok / 1_000_000 * prices["output"])
            costs.append(f"${cost:.4f}")
        print(f"  {model:<35}" + "".join(str(c).rjust(col_w) for c in costs))

    print()
    print("  (Cost is per 30-turn conversation. Multiply by # of conversations.)")


def print_storage_estimate(analyses: dict, turns_per_conv: int = 30):
    """Print rough storage overhead estimates."""
    # Embedding: ChromaDB stores text + float32 vector (1536 dims = 6KB per entry)
    BYTES_PER_EMBEDDING = 1536 * 4  # float32
    AVG_TURN_TEXT_BYTES = 800       # ~200 words user+assistant combined

    print("\n" + "=" * 60)
    print("  STORAGE OVERHEAD (per 30-turn conversation, estimated)")
    print("=" * 60)

    entries = {
        "raw_llm": {
            "description": "RAM only (sliding window)",
            "extra_bytes": 0,
        },
        "rag": {
            "description": "ChromaDB: all turns embedded",
            "extra_bytes": turns_per_conv * (BYTES_PER_EMBEDDING + AVG_TURN_TEXT_BYTES),
        },
        "rag_summary": {
            "description": "ChromaDB: all turns + rolling summary text",
            "extra_bytes": turns_per_conv * (BYTES_PER_EMBEDDING + AVG_TURN_TEXT_BYTES) + 2000,
        },
        "hiermem": {
            "description": "ChromaDB (L2/L3) + L0/L1 text + constraint store",
            # L2/L3: ~1 entry per 2 turns on average
            # L0/L1: text summaries, ~200 bytes per segment × 4 segments
            "extra_bytes": (turns_per_conv // 2) * (BYTES_PER_EMBEDDING + AVG_TURN_TEXT_BYTES)
                           + 4 * 400   # L0/L1 summaries
                           + 200,      # constraint store
        },
    }

    for sys_name, info in entries.items():
        kb = info["extra_bytes"] / 1024
        print(f"  {sys_name:<14} {kb:>7.1f} KB   {info['description']}")

    print()
    print("  Note: ChromaDB also uses disk for HNSW index (~2x raw size).")
    print("  HierMem uses ~50% less vector storage than RAG (fewer embeddings).")


def main():
    parser = argparse.ArgumentParser(description="Analyze compute cost from benchmark results")
    parser.add_argument("results_dir", help="Path to benchmark results directory")
    parser.add_argument("--costs", action="store_true", help="Show API cost projection table")
    parser.add_argument("--storage", action="store_true", help="Show storage overhead estimates")
    args = parser.parse_args()

    results_path = Path(args.results_dir) / "results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    print(f"Loaded results from: {results_path}")
    print(f"Systems found: {list(results.keys())}")

    analyses = {}
    for system_name, system_results in results.items():
        a = analyze_system(system_name, system_results)
        if a:
            analyses[system_name] = a

    print_comparison_table(analyses)

    if args.costs:
        print_cost_projection(analyses)

    if args.storage:
        print_storage_estimate(analyses)

    # Always show storage (it's cheap to compute and useful)
    if not args.storage:
        print_storage_estimate(analyses)

    print()


if __name__ == "__main__":
    main()

"""
Unified research metrics pipeline.

Purpose:
1. Read a benchmark run directory containing results.json and *_detailed.json.
2. Reorganize outputs into dataset-centric folders.
3. Create/consume LAAJ (LLM-as-a-Judge) files.
4. Compute research metrics (cost, latency, context, degradation proxies).
5. Generate publication-ready visualizations without zero-score artifacts.

Usage:
  python -m eval.research_metrics --run-dir results/raw/benchmarks/qwen14b_arch_b

Optional:
  --arch qwen14b_arch_b
  --judge-model gpt-4.1

Canonical post-processing command (replaces legacy tools):
  python -m eval.research_metrics --run-dir <dir> --arch <name> --judge-model gpt-4.1

Supersedes (kept for backward compatibility, do not delete yet):
  eval/visualize.py          — python eval/visualize.py --results <dir>
  eval/cost_analysis.py      — python eval/cost_analysis.py --results <dir>
  generate_visualizations.py — python generate_visualizations.py
  generate_cost_visualizations.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, variance
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


SYSTEM_ORDER = ["hiermem", "raw_llm", "rag", "rag_summary", "memgpt_style"]
SYSTEM_LABELS = {
    "hiermem": "HierMem",
    "raw_llm": "Raw LLM",
    "rag": "RAG",
    "rag_summary": "RAG Summary",
    "memgpt_style": "MemGPT-style",
}
SYSTEM_COLORS = {
    "hiermem": "#1f9d55",
    "raw_llm": "#e4572e",
    "rag": "#3a86ff",
    "rag_summary": "#ff9f1c",
    "memgpt_style": "#8338ec",
}

# Relative unit pricing (typical API market rates for Qwen2.5 models).
COST_PER_1M_TOKENS = {
    "main": 0.40,      # Qwen2.5-14B
    "curator": 0.10,   # Qwen2.5-3B
    "embedding": 0.05, # Standard embeddings
}

# Relative retrieval-op pricing (vector search requests/chunk fetches).
COST_PER_1K_RETRIEVAL_OPS = {
    "default": 0.05,
}

# Observed runtime cost proxy for local/cloud GPU execution.
# Override with env, e.g. GPU_HOURLY_USD=0.77
GPU_HOURLY_USD = float(os.getenv("GPU_HOURLY_USD", "0.77"))


@dataclass
class TurnCost:
    main_tokens: float
    curator_tokens: float
    embedding_tokens: float

    @property
    def total_tokens(self) -> float:
        return self.main_tokens + self.curator_tokens + self.embedding_tokens

    @property
    def total_cost_units(self) -> float:
        return (
            (self.main_tokens / 1_000_000.0) * COST_PER_1M_TOKENS["main"]
            + (self.curator_tokens / 1_000_000.0) * COST_PER_1M_TOKENS["curator"]
            + (self.embedding_tokens / 1_000_000.0) * COST_PER_1M_TOKENS["embedding"]
        )


def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _dataset_key(conversation: Dict[str, Any]) -> str:
    source = conversation.get("source_file")
    if source:
        return Path(source).stem
    cid = conversation.get("conversation_id", "unknown")
    if "__" in cid:
        return cid.split("__", 1)[0]
    return cid


def _iter_turn_logs(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    return conversation.get("turn_logs", [])


def _estimate_completion_tokens(response_text: str) -> int:
    if not response_text:
        return 0
    return max(1, len(response_text) // 4)


def _estimate_turn_cost(system_name: str, turn_log: Dict[str, Any]) -> TurnCost:
    """Compute token cost for one turn.

    For HierMem: uses REAL curator + postproc tokens when available (new pipeline),
    falls back to estimates for older result files that predate telemetry.
    For baselines: main tokens only (+ embedding for RAG systems).

    All values marked as ESTIMATED when real data unavailable.
    """
    pipeline = turn_log.get("pipeline_details") or {}
    context_tokens = pipeline.get("context_tokens_used", 0) or turn_log.get("context_tokens", 0) or 0

    # Main tokens: assembled context + approximate completion
    completion = _estimate_completion_tokens(turn_log.get("response", ""))
    main_tokens = max(0, int(context_tokens)) + completion

    # Curator tokens — prefer REAL measurements, fall back to estimates
    curator_tokens = 0
    if system_name == "hiermem":
        real_curator = pipeline.get("curator_total_tokens", 0)
        if real_curator and real_curator > 0:
            # Real measurement from pipeline telemetry
            curator_tokens = int(real_curator)
        else:
            # Estimate from strategy (older result files)
            strategy = (pipeline.get("curator_strategy") or "").upper()
            if strategy == "HYBRID":
                curator_tokens = max(32, int(main_tokens * 0.10))
            elif strategy and strategy != "NONE":
                curator_tokens = max(16, int(main_tokens * 0.04))

        # Add post-processor overhead (summarization + constraint extraction)
        # These are real 3b model calls that have a cost
        postproc_tokens = (
            pipeline.get("postproc_summary_tokens", 0)
            + pipeline.get("postproc_extract_tokens", 0)
            + pipeline.get("violation_retry_tokens", 0)
        )
        if postproc_tokens and postproc_tokens > 0:
            curator_tokens += int(postproc_tokens)
        else:
            # Estimate: summarizer + extractor ≈ 200 tokens per turn
            curator_tokens += 200

    embedding_tokens = 0
    if system_name in ("rag", "rag_summary"):
        queries = pipeline.get("semantic_queries") or []
        embedding_requests = int(pipeline.get("embedding_requests", 0) or 0)
        # Backward compatibility for old logs that only had vector_queries.
        legacy_vector = int(pipeline.get("vector_queries", 0) or 0)
        if queries:
            embedding_tokens = 120 * len(queries)
        elif embedding_requests > 0:
            embedding_tokens = 120 * embedding_requests
        elif legacy_vector > 0:
            embedding_tokens = 120

    return TurnCost(
        main_tokens=float(main_tokens),
        curator_tokens=float(curator_tokens),
        embedding_tokens=float(embedding_tokens),
    )


def _estimate_retrieval_ops(system_name: str, turn_log: Dict[str, Any]) -> int:
    """Estimate retrieval operations from pipeline telemetry.

    Retrieval ops are lightweight compute proxies for vector calls and chunk fetches.
    """
    pipeline = turn_log.get("pipeline_details") or {}

    if system_name == "hiermem":
        semantic_q = len(pipeline.get("semantic_queries") or [])
        seg_fetch = len(pipeline.get("segments_fetched") or [])
        full_turn_fetch = len(pipeline.get("fetch_full_turns") or [])
        return semantic_q + seg_fetch + full_turn_fetch

    if system_name in ("rag", "rag_summary"):
        retrieval_queries = int(pipeline.get("retrieval_query_count", 0) or 0)
        retrieved_chunks = int(pipeline.get("retrieved_chunks_count", 0) or 0)
        # Backward compatibility for older logs.
        if retrieved_chunks == 0 and "vector_queries" in pipeline:
            retrieved_chunks = int(pipeline.get("vector_queries", 0) or 0)
        return retrieval_queries + retrieved_chunks

    return 0


def _retrieval_ops_to_cost_units(retrieval_ops: int) -> float:
    return (float(retrieval_ops) / 1000.0) * COST_PER_1K_RETRIEVAL_OPS["default"]


def _load_results(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    results_file = run_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Missing results file: {results_file}")
    with results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("results.json must be a dict keyed by system name")
    return data


def _group_by_dataset(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for system_name, conversations in results.items():
        for conv in conversations:
            dkey = _dataset_key(conv)
            grouped[dkey][system_name].append(conv)
    return grouped


def _collect_checkpoints(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    checkpoints = conv.get("checkpoints", [])
    dedup = {}
    for cp in checkpoints:
        # Support both legacy and SE checkpoint keys.
        turn = cp.get("turn")
        if turn is None:
            turn = cp.get("checkpoint_turn")
        if turn is None:
            continue
        dedup[turn] = cp
    return [dedup[t] for t in sorted(dedup.keys())]


def _load_laaj_scores(laaj_path: Path) -> Dict[Tuple[str, str, int], float]:
    """Load judge scores from a laaj.json file.

    Accepts TWO schemas:

    Schema A — flat template (produced by _ensure_laaj_template):
        { "judgments": [{ "system": "hiermem", "conversation_id": "...",
                          "turn": 19, "weighted_score": 9.85, ... }] }

    Schema B — LAAJ prompt output (produced by GPT-4.1 with the evaluation prompt):
        { "system_evaluations": { "system_a": { "checkpoints": [...] } },
          "system_identity_reveal": { "system_a": "hiermem", ... },
          "evaluation_metadata": { "benchmark": "sql_databases_parameterized_orm" } }

    For Schema B, checkpoint turns are normalized using the formula:
        dataset_turn = 2 * normalized_turn - 1
    where normalized_turn is the Nth exchange number (10, 20, 30...).
    This maps exchange 10 -> turn 19, exchange 20 -> turn 39, etc.

    Returns dict keyed by (system_name, conversation_id, dataset_turn) -> weighted_score.
    """
    if not laaj_path.exists():
        return {}
    with laaj_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        return {}

    scores: Dict[Tuple[str, str, int], float] = {}

    # ── Schema B: LAAJ prompt output (system_evaluations + system_identity_reveal) ──
    if "system_evaluations" in payload and "system_identity_reveal" in payload:
        identity: Dict[str, str] = {
            k: v for k, v in payload["system_identity_reveal"].items()
            if k != "ranked_reveal"
        }
        meta = payload.get("evaluation_metadata", {})
        # Conversation ID can be stored under different metadata keys.
        conversation_id = str(
            meta.get("benchmark")
            or meta.get("conversation_id")
            or meta.get("dataset_id")
            or "unknown"
        )
        sys_evals = payload["system_evaluations"]

        for sys_key, sys_data in sys_evals.items():
            real_name = identity.get(sys_key, sys_key)
            for cp in sys_data.get("checkpoints", []):
                cp_turn = cp.get("checkpoint_turn")
                if cp_turn is not None:
                    # Modern Schema-B outputs provide canonical global checkpoint turns.
                    dataset_turn = int(cp_turn)
                else:
                # LAAJ prompt stores exchange number in "turn" (10, 20, 30...)
                # normalized_turn may also be present — use whichever gives an integer
                    raw_turn = cp.get("turn")
                    norm_turn = cp.get("normalized_turn")

                    # Legacy fallback: exchange-number style turns -> odd dataset turns.
                    conv_len = int(meta.get("conversation_length", 50))
                    if raw_turn is not None and raw_turn % 2 == 0 and raw_turn <= conv_len:
                        dataset_turn = 2 * raw_turn - 1
                    elif raw_turn is not None and raw_turn % 2 == 1:
                        dataset_turn = raw_turn
                    elif norm_turn is not None:
                        dataset_turn = 2 * norm_turn - 1
                    else:
                        continue

                # Extract weighted_score — prefer direct field, compute from sub-scores if absent
                ws = cp.get("weighted_score")
                if ws is None:
                    sub = cp.get("sub_scores", {})
                    ca = sub.get("constraint_adherence")
                    rq = sub.get("response_quality")
                    cc = sub.get("conversational_coherence")
                    if ca is not None and rq is not None and cc is not None:
                        ws = round(0.5 * float(ca) + 0.3 * float(rq) + 0.2 * float(cc), 4)

                if ws is None:
                    continue

                score_val = float(ws)
                scores[(real_name, conversation_id, int(dataset_turn))] = score_val
                # Fallback key for schema-B files that do not carry run-benchmark conversation_id.
                scores[(real_name, "__any__", int(dataset_turn))] = score_val

        return scores

    # ── Schema A: flat template (judgments list) ──
    judgments = payload.get("judgments", [])
    for j in judgments:
        try:
            system = str(j.get("system"))
            convo = str(j.get("conversation_id"))
            turn = int(j.get("turn"))
        except Exception:
            continue

        ws = j.get("weighted_score")
        if ws is None:
            ca = j.get("constraint_adherence")
            rq = j.get("response_quality")
            cc = j.get("conversational_coherence")
            if ca is not None and rq is not None and cc is not None:
                ws = 0.5 * float(ca) + 0.3 * float(rq) + 0.2 * float(cc)

        if ws is None:
            continue
        scores[(system, convo, turn)] = float(ws)

    return scores


def _judge_status(expected: int, present: int) -> str:
    """Classify completeness of LAAJ judging for a dataset.

    Returns one of:
      "no_checkpoints"   – dataset has no evaluation checkpoints
      "pending_judge"    – checkpoints exist but none have been scored yet
      "partial_judge"    – some checkpoints scored, some still null
      "ready_for_paper"  – all expected checkpoints have been scored
    """
    if expected == 0:
        return "no_checkpoints"
    if present == 0:
        return "pending_judge"
    if present < expected:
        return "partial_judge"
    return "ready_for_paper"


def _build_metrics_for_dataset(
    dataset_name: str,
    dataset_system_results: Dict[str, List[Dict[str, Any]]],
    laaj_scores: Dict[Tuple[str, str, int], float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dataset": dataset_name,
        "systems": {},
        "coverage": {
            "judge_scores_present": 0,
            "judge_scores_expected": 0,
        },
    }

    for system_name, convos in dataset_system_results.items():
        turn_latencies: List[float] = []
        turn_costs: List[float] = []
        retrieval_ops: List[float] = []
        retrieval_costs: List[float] = []
        runtime_costs_usd: List[float] = []
        main_tokens: List[float] = []
        curator_tokens: List[float] = []
        embedding_tokens: List[float] = []
        context_util: List[float] = []
        judge_scores: List[float] = []
        checkpoint_expected = 0

        # Degradation tracking: per-turn judge score lists and violation detection
        degradation_by_turn: Dict[int, List[float]] = {}
        first_violation_turn: Optional[int] = None
        # A weighted_score < 5.0 on the [1,10] scale signals a constraint violation
        VIOLATION_THRESHOLD = 5.0

        # Track per-turn context sent for accumulated context analysis
        per_turn_context_sent: List[float] = []
        accumulated_context_total: float = 0.0

        for conv in convos:
            cid = conv.get("conversation_id", "unknown")
            turns = _iter_turn_logs(conv)
            budget = conv.get("config", {}).get("context_budget", 8192)

            # Compute accumulated raw conversation tokens for this conversation
            conv_accumulated = 0.0
            for t in turns:
                user_text = t.get("user", "")
                resp_text = t.get("response", "")
                conv_accumulated += max(1, len(user_text) // 4) + max(1, len(resp_text) // 4)
            accumulated_context_total = max(accumulated_context_total, conv_accumulated)

            for t in turns:
                lat = t.get("latency_seconds")
                if isinstance(lat, (int, float)):
                    turn_latencies.append(float(lat))
                    runtime_costs_usd.append((float(lat) / 3600.0) * GPU_HOURLY_USD)

                tc = _estimate_turn_cost(system_name, t)
                # NOTE: all cost figures are ESTIMATED from token counts, not observed
                turn_costs.append(tc.total_cost_units)
                rops = _estimate_retrieval_ops(system_name, t)
                retrieval_ops.append(float(rops))
                retrieval_costs.append(_retrieval_ops_to_cost_units(rops))
                main_tokens.append(tc.main_tokens)
                curator_tokens.append(tc.curator_tokens)
                embedding_tokens.append(tc.embedding_tokens)

                pipeline = t.get("pipeline_details") or {}
                ctx_used = pipeline.get("context_tokens_used", 0) or t.get("context_tokens", 0) or 0
                per_turn_context_sent.append(float(ctx_used))
                if budget and ctx_used:
                    context_util.append(min(1.0, float(ctx_used) / float(budget)))

            for cp in _collect_checkpoints(conv):
                checkpoint_expected += 1
                cp_turn = cp.get("turn")
                if cp_turn is None:
                    cp_turn = cp.get("checkpoint_turn", 0)
                cp_turn = int(cp_turn)
                key = (system_name, cid, cp_turn)
                fallback_key = (system_name, "__any__", cp_turn)
                if key in laaj_scores or fallback_key in laaj_scores:
                    score = laaj_scores.get(key, laaj_scores.get(fallback_key))
                    judge_scores.append(score)
                    degradation_by_turn.setdefault(cp_turn, []).append(score)
                    # Track first turn where a violation was judged
                    if score < VIOLATION_THRESHOLD and first_violation_turn is None:
                        first_violation_turn = cp_turn

        out["coverage"]["judge_scores_expected"] += checkpoint_expected
        out["coverage"]["judge_scores_present"] += len(judge_scores)

        # --- Aggregate scores — never default to 0 when absent ---
        avg_score: Optional[float] = mean(judge_scores) if judge_scores else None
        total_token_cost = float(sum(turn_costs))
        total_retrieval_cost = float(sum(retrieval_costs))
        total_compute_cost = total_token_cost + total_retrieval_cost
        total_runtime_cost_usd = float(sum(runtime_costs_usd))

        # cost_per_quality_point is undefined (None) when judge scores are missing.
        # Never substitute 0 or a sentinel like 999 — callers must handle None.
        cqp_token: Optional[float] = (
            (total_token_cost / avg_score) if (avg_score and avg_score > 0) else None
        )
        cqp_compute: Optional[float] = (
            (total_compute_cost / avg_score) if (avg_score and avg_score > 0) else None
        )
        cqp_runtime_usd: Optional[float] = (
            (total_runtime_cost_usd / avg_score) if (avg_score and avg_score > 0) else None
        )

        # Degradation curve: average LAAJ score at each checkpoint turn
        degradation_curve: Dict[int, float] = {
            t: round(mean(scores), 4)
            for t, scores in sorted(degradation_by_turn.items())
        }

        # Constraint survival rate: fraction of judged checkpoints with score >= threshold
        all_scored = [s for scores in degradation_by_turn.values() for s in scores]
        survival_rate: Optional[float] = (
            round(sum(1 for s in all_scored if s >= VIOLATION_THRESHOLD) / len(all_scored), 4)
            if all_scored else None
        )

        # Token totals for share calculation — use 1.0 floor to avoid division by zero
        total_tokens_all = max(
            1.0,
            sum(main_tokens) + sum(curator_tokens) + sum(embedding_tokens),
        )

        out["systems"][system_name] = {
            "num_conversations": len(convos),
            # --- Latency (OBSERVED from wall-clock) ---
            "avg_latency_seconds": round(mean(turn_latencies), 3) if turn_latencies else None,
            # --- Cost (ESTIMATED from token counts) ---
            "avg_cost_per_turn_units": round(mean(turn_costs), 8) if turn_costs else None,
            "total_cost_units": round(total_token_cost, 8),
            "avg_retrieval_ops_per_turn": round(mean(retrieval_ops), 4) if retrieval_ops else None,
            "avg_retrieval_cost_per_turn_units": round(mean(retrieval_costs), 8) if retrieval_costs else None,
            "total_retrieval_cost_units": round(total_retrieval_cost, 8),
            "avg_compute_cost_per_turn_units": round((total_compute_cost / max(len(turn_costs), 1)), 8)
            if turn_costs else None,
            "total_compute_cost_units": round(total_compute_cost, 8),
            "avg_runtime_cost_per_turn_usd": round(mean(runtime_costs_usd), 8) if runtime_costs_usd else None,
            "total_runtime_cost_usd": round(total_runtime_cost_usd, 8),
            "avg_main_tokens": round(mean(main_tokens), 2) if main_tokens else None,
            "avg_curator_tokens": round(mean(curator_tokens), 2) if curator_tokens else None,
            "avg_embedding_tokens": round(mean(embedding_tokens), 2) if embedding_tokens else None,
            "curator_token_share": round(sum(curator_tokens) / total_tokens_all, 5),
            "context_utilization_pct": round(100.0 * mean(context_util), 2) if context_util else None,
            # --- Judge scores (OBSERVED, only present when laaj.json is filled) ---
            "judge_score_avg": round(avg_score, 4) if avg_score is not None else None,
            "judge_samples": len(judge_scores),
            # cost_per_quality_point remains backward-compatible but now reflects compute cost
            # (token + retrieval) for fairer system comparisons.
            "cost_per_quality_point": round(cqp_compute, 8) if cqp_compute is not None else None,
            "token_cost_per_quality_point": round(cqp_token, 8) if cqp_token is not None else None,
            "runtime_cost_per_quality_point_usd": round(cqp_runtime_usd, 8) if cqp_runtime_usd is not None else None,
            # --- Degradation & survival (derived from LAAJ, None when no scores) ---
            "first_violation_turn": first_violation_turn,
            "constraint_survival_rate": survival_rate,
            "degradation_curve": degradation_curve,
            # --- NEW: Accumulated context & compression metrics ---
            "total_accumulated_tokens": round(accumulated_context_total, 0),
            "avg_context_sent_per_turn": round(mean(per_turn_context_sent), 2) if per_turn_context_sent else None,
            "compression_ratio": round(
                accumulated_context_total / max(mean(per_turn_context_sent), 1), 2
            ) if per_turn_context_sent and accumulated_context_total > 0 else None,
            "token_savings_vs_full_history_pct": round(
                100.0 * (1.0 - mean(per_turn_context_sent) / max(accumulated_context_total, 1)), 1
            ) if per_turn_context_sent and accumulated_context_total > 0 else None,
        }

    # --- Dataset-level status and confidence flags (written after all systems) ---
    out["dataset_status"] = _judge_status(
        out["coverage"]["judge_scores_expected"],
        out["coverage"]["judge_scores_present"],
    )
    out["confidence_flags"] = {
        # Latency comes from wall-clock measurements in run_benchmark.py
        "latency_is_observed": True,
        # All cost figures are derived from token counts, not billed invoices
        "cost_is_estimated": True,
        # Token breakdown uses heuristics (curator ≈ 4-10% of main, embedding ≈ 120 per query)
        "token_breakdown_is_estimated": True,
        # Retrieval ops pricing is a normalized proxy unless replaced with infra billing data
        "retrieval_cost_is_estimated": True,
        # Runtime cost derives from measured latency and configured hourly GPU price
        "runtime_cost_is_observed_latency_scaled": True,
        "gpu_hourly_usd": GPU_HOURLY_USD,
        # Judge scores are real LAAJ judgments only when status is partial or ready
        "judge_score_is_observed": out["dataset_status"] in ("partial_judge", "ready_for_paper"),
    }

    return out


def _write_dataset_payloads(dataset_dir: Path, dataset_system_results: Dict[str, List[Dict[str, Any]]]) -> None:
    for system_name, convos in dataset_system_results.items():
        out_path = dataset_dir / f"{system_name}_detailed.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(convos, f, ensure_ascii=False, indent=2)


def _write_summary_csv(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> None:
    csv_path = dataset_dir / "metrics_summary.csv"
    fields = [
        "system",
        "num_conversations",
        "judge_score_avg",
        "judge_samples",
        "avg_latency_seconds",
        "avg_cost_per_turn_units",
        "avg_retrieval_ops_per_turn",
        "avg_retrieval_cost_per_turn_units",
        "avg_compute_cost_per_turn_units",
        "total_cost_units",
        "total_retrieval_cost_units",
        "total_compute_cost_units",
        "avg_runtime_cost_per_turn_usd",
        "total_runtime_cost_usd",
        "cost_per_quality_point",
        "token_cost_per_quality_point",
        "runtime_cost_per_quality_point_usd",
        "avg_main_tokens",
        "avg_curator_tokens",
        "avg_embedding_tokens",
        "curator_token_share",
        "context_utilization_pct",
        # Degradation / survival fields (None when laaj.json not yet filled)
        "first_violation_turn",
        "constraint_survival_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        # extrasaction='ignore' so nested fields like degradation_curve
        # (which is a dict, not a scalar) are silently omitted from the flat CSV.
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for system_name in SYSTEM_ORDER:
            if system_name not in metrics_payload["systems"]:
                continue
            row = {"system": system_name}
            row.update(metrics_payload["systems"][system_name])
            writer.writerow(row)


def _plot_quality_cost(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> bool:
    """Scatter plot of cost vs LAAJ quality score.

    Returns True if the chart was written, False if skipped (no systems
    have both a judge score and cost data). The caller records this in
    visualization_status.json so paper authors know why the file is absent.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted = 0
    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        x = row.get("avg_cost_per_turn_units")
        y = row.get("judge_score_avg")
        if x is None or y is None:
            # Skip silently — judge scores not yet filled in laaj.json
            continue
        plotted += 1
        ax.scatter(
            x,
            y,
            s=180,
            color=SYSTEM_COLORS.get(system_name, "#888888"),
            edgecolors="black",
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )
        ax.annotate(
            SYSTEM_LABELS.get(system_name, system_name),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
        )

    if plotted == 0:
        plt.close(fig)
        return False  # caller records as skipped

    ax.set_xlabel("Average Cost per Turn (relative units)", fontsize=12)
    ax.set_ylabel("LAAJ Score (1–10)", fontsize=12)
    ax.set_title("Cost vs LAAJ Quality", fontsize=13)
    ax.grid(alpha=0.25)
    # y-axis is always [1, 10] when judge data is present — never starts at 0
    ax.set_ylim(1.0, 10.0)
    fig.tight_layout()
    fig.savefig(dataset_dir / "cost_vs_quality.png", dpi=220)
    plt.close(fig)
    return True


def _plot_latency(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> bool:
    """Bar chart of avg latency per turn. Returns True if written, False if skipped."""
    labels = []
    values = []
    colors = []
    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        val = row.get("avg_latency_seconds")
        if val is None:
            continue
        labels.append(SYSTEM_LABELS.get(system_name, system_name))
        values.append(val)
        colors.append(SYSTEM_COLORS.get(system_name, "#888888"))

    if not values:
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylabel("Avg Latency per Turn (s)", fontsize=12)
    ax.set_title("Latency Comparison", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(dataset_dir / "latency_comparison.png", dpi=220)
    plt.close(fig)
    return True


def _plot_token_breakdown(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> bool:
    """Stacked bar chart of avg tokens per turn by component. Returns True if written."""
    systems = []
    main_vals = []
    cur_vals = []
    emb_vals = []

    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        # Use 0.0 only for chart rendering (a bar of 0 height is correct here);
        # the underlying JSON stores None to distinguish missing from zero.
        systems.append(SYSTEM_LABELS.get(system_name, system_name))
        main_vals.append(float(row.get("avg_main_tokens") or 0.0))
        cur_vals.append(float(row.get("avg_curator_tokens") or 0.0))
        emb_vals.append(float(row.get("avg_embedding_tokens") or 0.0))

    if not systems:
        return False

    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, main_vals, label="Main LLM", color="#3a86ff")
    ax.bar(x, cur_vals, bottom=main_vals, label="Curator", color="#1f9d55")
    ax.bar(x, emb_vals, bottom=np.array(main_vals) + np.array(cur_vals), label="Embedding", color="#ff9f1c")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20)
    ax.set_ylabel("Avg Tokens per Turn (estimated)", fontsize=12)
    ax.set_title("Token Composition by System", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(dataset_dir / "token_composition.png", dpi=220)
    plt.close(fig)
    return True


def _plot_degradation_curve(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> bool:
    """Line plot of LAAJ score at each checkpoint turn per system.

    THE most important chart — shows exactly when each architecture starts
    forgetting constraints.  Returns True if written.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        curve = row.get("degradation_curve") or {}
        if not curve:
            continue
        turns = sorted(curve.keys(), key=int)
        scores = [curve[t] for t in turns]
        turns_int = [int(t) for t in turns]
        ax.plot(
            turns_int, scores, marker="o", linewidth=2.2,
            label=SYSTEM_LABELS.get(system_name, system_name),
            color=SYSTEM_COLORS.get(system_name, "#888888"),
            alpha=0.9,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel("Checkpoint Turn", fontsize=12)
    ax.set_ylabel("LAAJ Score (1–10)", fontsize=12)
    ax.set_title("Constraint Degradation Over Conversation", fontsize=13)
    ax.set_ylim(0.5, 10.5)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dataset_dir / "degradation_curve.png", dpi=220)
    plt.close(fig)
    return True


def _plot_cumulative_token_cost(
    dataset_dir: Path,
    results: Dict[str, List[Dict[str, Any]]],
) -> bool:
    """Line plot of cumulative main-LLM tokens over conversation turns.

    Proves HierMem's O(1) flat scaling vs Raw-LLM's O(n²) growth.
    Uses raw turn_logs directly (not aggregated metrics).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for system_name in SYSTEM_ORDER:
        convos = results.get(system_name, [])
        if not convos:
            continue
        # Use first conversation for the per-turn curve
        turn_logs = convos[0].get("turn_logs", []) if convos else []
        if not turn_logs:
            continue
        cumulative = []
        running = 0.0
        turn_nums = []
        for tl in turn_logs:
            tc = _estimate_turn_cost(system_name, tl)
            running += tc.main_tokens
            cumulative.append(running)
            turn_nums.append(tl.get("turn", len(cumulative)))
        ax.plot(
            turn_nums, cumulative, linewidth=2.2,
            label=SYSTEM_LABELS.get(system_name, system_name),
            color=SYSTEM_COLORS.get(system_name, "#888888"),
            alpha=0.9,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Cumulative Main-LLM Tokens", fontsize=12)
    ax.set_title("Cumulative Token Cost Over Conversation", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dataset_dir / "cumulative_token_cost.png", dpi=220)
    plt.close(fig)
    return True


def _plot_context_utilization_heatmap(
    dataset_dir: Path,
    results: Dict[str, List[Dict[str, Any]]],
) -> bool:
    """Heatmap of context_tokens / budget at each turn for each system."""
    system_names = [s for s in SYSTEM_ORDER if s in results and results[s]]
    if not system_names:
        return False

    # Build matrix: rows=systems, cols=turns
    max_turns = 0
    data_rows = []
    for sn in system_names:
        turns = results[sn][0].get("turn_logs", []) if results[sn] else []
        budget = results[sn][0].get("config", {}).get("context_budget", 8192) if results[sn] else 8192
        row = []
        for tl in turns:
            pipeline = tl.get("pipeline_details") or {}
            ctx = pipeline.get("context_tokens_used", 0) or tl.get("context_tokens", 0) or 0
            row.append(min(1.0, float(ctx) / max(float(budget), 1.0)))
        data_rows.append(row)
        max_turns = max(max_turns, len(row))

    if max_turns == 0:
        return False

    # Pad shorter rows
    for row in data_rows:
        while len(row) < max_turns:
            row.append(0.0)

    matrix = np.array(data_rows)
    fig, ax = plt.subplots(figsize=(14, max(3, len(system_names) * 0.8 + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1.0)
    ax.set_yticks(range(len(system_names)))
    ax.set_yticklabels([SYSTEM_LABELS.get(s, s) for s in system_names])
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_title("Context Window Utilization (% of Budget)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Utilization", shrink=0.8)
    fig.tight_layout()
    fig.savefig(dataset_dir / "context_utilization_heatmap.png", dpi=220)
    plt.close(fig)
    return True


def _plot_survival_rate(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> bool:
    """Bar chart of constraint survival rate (% checkpoints with score ≥ 5.0)."""
    labels = []
    values = []
    colors = []
    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        sr = row.get("constraint_survival_rate")
        if sr is None:
            continue
        labels.append(SYSTEM_LABELS.get(system_name, system_name))
        values.append(sr * 100.0)
        colors.append(SYSTEM_COLORS.get(system_name, "#888888"))

    if not values:
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Constraint Survival Rate (%)", fontsize=12)
    ax.set_title("Checkpoints Passing Constraint Threshold (≥ 5.0/10)", fontsize=13)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(dataset_dir / "constraint_survival.png", dpi=220)
    plt.close(fig)
    return True


def _plot_cost_per_quality(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> bool:
    """Bar chart of cost-per-quality-point (lower = more efficient)."""
    labels = []
    values = []
    colors = []
    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        cpq = row.get("cost_per_quality_point")
        if cpq is None:
            continue
        labels.append(SYSTEM_LABELS.get(system_name, system_name))
        values.append(cpq)
        colors.append(SYSTEM_COLORS.get(system_name, "#888888"))

    if not values:
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Cost per Quality Point (lower = better)", fontsize=12)
    ax.set_title("Cost Efficiency: Token Cost ÷ LAAJ Score", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(dataset_dir / "cost_per_quality.png", dpi=220)
    plt.close(fig)
    return True


def _plot_curator_overhead(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> bool:
    """Pie chart of HierMem's token breakdown: Main vs Curator vs PostProc."""
    row = metrics_payload["systems"].get("hiermem")
    if not row:
        return False
    main_t = float(row.get("avg_main_tokens") or 0)
    cur_t = float(row.get("avg_curator_tokens") or 0)
    emb_t = float(row.get("avg_embedding_tokens") or 0)
    total = main_t + cur_t + emb_t
    if total <= 0:
        return False

    slices = []
    slice_labels = []
    slice_colors = []
    if main_t > 0:
        slices.append(main_t)
        slice_labels.append(f"Main LLM\n({main_t:.0f} tok)")
        slice_colors.append("#3a86ff")
    if cur_t > 0:
        slices.append(cur_t)
        slice_labels.append(f"Curator + PostProc\n({cur_t:.0f} tok)")
        slice_colors.append("#1f9d55")
    if emb_t > 0:
        slices.append(emb_t)
        slice_labels.append(f"Embedding\n({emb_t:.0f} tok)")
        slice_colors.append("#ff9f1c")

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        slices, labels=slice_labels, colors=slice_colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("HierMem Token Overhead Breakdown", fontsize=13)
    fig.tight_layout()
    fig.savefig(dataset_dir / "curator_overhead.png", dpi=220)
    plt.close(fig)
    return True


def _plot_context_pressure(
    dataset_dir: Path,
    results: Dict[str, List[Dict[str, Any]]],
) -> bool:
    """Line plot showing context tokens sent per turn vs total accumulated conversation.

    Demonstrates the compression ratio: the gap between what the conversation
    generated and what the system actually sends to the model. A 32k ceiling
    line shows the KV cache / VRAM limit.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    # Plot per-system context sent per turn
    for system_name in SYSTEM_ORDER:
        convos = results.get(system_name, [])
        if not convos:
            continue
        turn_logs = convos[0].get("turn_logs", []) if convos else []
        if not turn_logs:
            continue
        ctx_vals = []
        turn_nums = []
        for tl in turn_logs:
            pipeline = tl.get("pipeline_details") or {}
            ctx = pipeline.get("context_tokens_used", 0) or tl.get("context_tokens", 0) or 0
            ctx_vals.append(float(ctx))
            turn_nums.append(tl.get("turn", len(ctx_vals)))
        ax.plot(
            turn_nums, ctx_vals, linewidth=2.2,
            label=SYSTEM_LABELS.get(system_name, system_name),
            color=SYSTEM_COLORS.get(system_name, "#888888"),
            alpha=0.9,
        )
        plotted += 1

    # Plot accumulated context from any system (same conversation content)
    any_system = next((s for s in SYSTEM_ORDER if s in results and results[s]), None)
    if any_system:
        turn_logs = results[any_system][0].get("turn_logs", [])
        accumulated = 0.0
        accum_vals = []
        turn_nums_accum = []
        for tl in turn_logs:
            user_text = tl.get("user", "")
            resp_text = tl.get("response", "")
            accumulated += max(1, len(user_text) // 4) + max(1, len(resp_text) // 4)
            accum_vals.append(accumulated)
            turn_nums_accum.append(tl.get("turn", len(accum_vals)))
        ax.plot(
            turn_nums_accum, accum_vals, linewidth=2.5, linestyle="--",
            color="#666666", alpha=0.8, label="Total Conversation",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    # 32k ceiling line
    ax.axhline(y=32768, color="#dc2626", linestyle=":", linewidth=2.0,
               label="32k Context Limit", alpha=0.7)

    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("Tokens", fontsize=12)
    ax.set_title("Context Pressure: Sent vs Total Generated", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dataset_dir / "context_pressure.png", dpi=220)
    plt.close(fig)
    return True


def _write_visualization_status(
    dataset_dir: Path,
    metrics_payload: Dict[str, Any],
    charts_written: List[str],
    charts_skipped: List[str],
) -> None:
    """Write visualization_status.json explaining which charts exist and why others were skipped."""
    skip_reasons: Dict[str, str] = {}
    _skip_map = {
        "cost_vs_quality.png": "No systems had both judge_score_avg and cost data. Fill laaj.json and re-run.",
        "latency_comparison.png": "No avg_latency_seconds data found for any system.",
        "token_composition.png": "No systems found in this dataset.",
        "degradation_curve.png": "No LAAJ degradation_curve data. Fill laaj.json and re-run.",
        "cumulative_token_cost.png": "No turn_logs found to plot cumulative cost.",
        "context_utilization_heatmap.png": "No turn_logs found to plot context utilization.",
        "constraint_survival.png": "No constraint_survival_rate data. Fill laaj.json and re-run.",
        "cost_per_quality.png": "No cost_per_quality_point data. Fill laaj.json and re-run.",
        "curator_overhead.png": "HierMem system not present or no token data.",
    }
    for chart_name in charts_skipped:
        if chart_name in _skip_map:
            skip_reasons[chart_name] = _skip_map[chart_name]

    payload = {
        "dataset_status": metrics_payload.get("dataset_status", "unknown"),
        "charts_written": sorted(charts_written),
        "charts_skipped": sorted(charts_skipped),
        "skip_reasons": skip_reasons,
    }
    (dataset_dir / "visualization_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _build_arch_aggregation(
    arch_name: str,
    source_run_dir: str,
    all_dataset_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-dataset metrics into arch-level statistics per system.

    For each system computes:
      - datasets_included      : int, number of datasets where system appears
      - mean_judge_score       : float | None, mean LAAJ score across datasets
      - score_variance         : float | None, variance (None when <2 data points)
      - mean_cost_per_turn     : float | None, mean cost per turn across datasets
      - cost_variance          : float | None
    """
    system_scores: Dict[str, List[float]] = {}
    system_costs: Dict[str, List[float]] = {}
    system_compute_costs: Dict[str, List[float]] = {}
    system_dataset_count: Dict[str, int] = {}

    for dataset_name, dmx in all_dataset_metrics.items():
        for sys_name, row in dmx.get("systems", {}).items():
            score = row.get("judge_score_avg")
            cost = row.get("avg_cost_per_turn_units")
            compute_cost = row.get("avg_compute_cost_per_turn_units")
            system_dataset_count[sys_name] = system_dataset_count.get(sys_name, 0) + 1
            if score is not None:
                system_scores.setdefault(sys_name, []).append(score)
            if cost is not None:
                system_costs.setdefault(sys_name, []).append(cost)
            if compute_cost is not None:
                system_compute_costs.setdefault(sys_name, []).append(compute_cost)

    agg_systems: Dict[str, Any] = {}
    all_sys = set(list(system_scores) + list(system_costs) + list(system_dataset_count))
    for sys_name in all_sys:
        scores = system_scores.get(sys_name, [])
        costs = system_costs.get(sys_name, [])
        compute_costs = system_compute_costs.get(sys_name, [])
        agg_systems[sys_name] = {
            "datasets_included": system_dataset_count.get(sys_name, 0),
            "mean_judge_score": round(mean(scores), 4) if scores else None,
            # variance() requires ≥2 points; guard explicitly
            "score_variance": round(variance(scores), 6) if len(scores) > 1 else None,
            "mean_cost_per_turn": round(mean(costs), 8) if costs else None,
            "cost_variance": round(variance(costs), 10) if len(costs) > 1 else None,
            "mean_compute_cost_per_turn": round(mean(compute_costs), 8) if compute_costs else None,
            "compute_cost_variance": round(variance(compute_costs), 10) if len(compute_costs) > 1 else None,
        }

    return {
        "metadata": {
            "arch": arch_name,
            "source_run_dir": source_run_dir,
            "datasets_count": len(all_dataset_metrics),
            "cost_units": {
                "tokens_per_1m": COST_PER_1M_TOKENS,
                "retrieval_ops_per_1k": COST_PER_1K_RETRIEVAL_OPS,
                "gpu_hourly_usd": GPU_HOURLY_USD,
            },
        },
        "systems": agg_systems,
        # Verbatim per-dataset breakdown nested here for one-stop reading
        "datasets": all_dataset_metrics,
    }


def process_single_dataset(run_dir: Path) -> Path:
    """Process one dataset folder containing results.json + optional laaj.json.

    All outputs are written INTO run_dir itself — no nested subfolder is created.

    Expected layout:
        run_dir/
          results.json               <- benchmark output (required)
          hiermem_detailed.json      <- written by run_benchmark, already there
          raw_llm_detailed.json
          rag_detailed.json
          rag_summary_detailed.json
          laaj.json                  <- you place this after GPT-4.1 judging (optional)

    Outputs written into run_dir:
          metrics_research.json
          metrics_summary.csv
          latency_comparison.png
          token_composition.png
          cost_vs_quality.png        <- only when laaj.json has scores
          visualization_status.json
    """
    results = _load_results(run_dir)
    grouped = _group_by_dataset(results)

    # Merge all conversations across groups (single-dataset folder has one group)
    all_system_results: Dict[str, List[Dict[str, Any]]] = {}
    dataset_name = run_dir.name
    for dkey, sys_results in grouped.items():
        for sys_name, convos in sys_results.items():
            all_system_results.setdefault(sys_name, []).extend(convos)
        dataset_name = dkey  # use the key derived from conversation_id / source_file

    laaj_path = run_dir / "laaj.json"
    laaj_scores = _load_laaj_scores(laaj_path)

    metrics_payload = _build_metrics_for_dataset(dataset_name, all_system_results, laaj_scores)

    with (run_dir / "metrics_research.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    _write_summary_csv(run_dir, metrics_payload)

    charts_written: List[str] = []
    charts_skipped: List[str] = []

    def _record(name: str, wrote: bool) -> None:
        (charts_written if wrote else charts_skipped).append(name)

    _record("cost_vs_quality.png",    _plot_quality_cost(run_dir, metrics_payload))
    _record("latency_comparison.png", _plot_latency(run_dir, metrics_payload))
    _record("token_composition.png",  _plot_token_breakdown(run_dir, metrics_payload))
    _record("degradation_curve.png",  _plot_degradation_curve(run_dir, metrics_payload))
    _record("cumulative_token_cost.png", _plot_cumulative_token_cost(run_dir, all_system_results))
    _record("context_utilization_heatmap.png", _plot_context_utilization_heatmap(run_dir, all_system_results))
    _record("constraint_survival.png",  _plot_survival_rate(run_dir, metrics_payload))
    _record("cost_per_quality.png",     _plot_cost_per_quality(run_dir, metrics_payload))
    _record("curator_overhead.png",     _plot_curator_overhead(run_dir, metrics_payload))
    _record("context_pressure.png",     _plot_context_pressure(run_dir, all_system_results))

    _write_visualization_status(run_dir, metrics_payload, charts_written, charts_skipped)

    status = metrics_payload.get("dataset_status", "unknown")
    print(f"[{status}] {dataset_name}")
    print(f"  charts written : {', '.join(charts_written) or 'none'}")
    print(f"  charts skipped : {', '.join(charts_skipped) or 'none'}")
    if laaj_scores:
        for sys_name, row in metrics_payload["systems"].items():
            print(f"  {sys_name:<14} judge={row.get('judge_score_avg')}  "
                  f"cost/turn={row.get('avg_cost_per_turn_units')}")
    else:
        print("  No laaj.json found — place it here and re-run to get judge metrics.")

    return run_dir


def process_aggregate(arch_dir: Path) -> Path:
    """Aggregate metrics_research.json from all dataset sub-folders into arch_metrics_research.json.

    Call this after all individual datasets have been processed.

    Layout:
        arch_dir/
          dataset_03_sql_databases/
            metrics_research.json    <- must exist first
          dataset_01_js_utils_final/
            metrics_research.json
          ...
          arch_metrics_research.json <- written here by this function
    """
    all_dataset_metrics: Dict[str, Dict[str, Any]] = {}

    for sub in sorted(arch_dir.iterdir()):
        if not sub.is_dir():
            continue
        mf = sub / "metrics_research.json"
        if not mf.exists():
            continue
        payload = json.loads(mf.read_text(encoding="utf-8"))
        dname = payload.get("dataset", sub.name)
        all_dataset_metrics[dname] = payload
        status = payload.get("dataset_status", "unknown")
        scores = {s: r.get("judge_score_avg") for s, r in payload.get("systems", {}).items()}
        print(f"  [{status}] {dname}: {scores}")

    if not all_dataset_metrics:
        print(f"No metrics_research.json files found in sub-folders of {arch_dir}")
        print("Run --run-dir on each dataset folder first.")
        return arch_dir

    arch_agg = _build_arch_aggregation(
        arch_name=arch_dir.name,
        source_run_dir=str(arch_dir),
        all_dataset_metrics=all_dataset_metrics,
    )
    out_path = arch_dir / "arch_metrics_research.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(arch_agg, f, ensure_ascii=False, indent=2)

    print(f"\nArch aggregation: {out_path}")
    print(f"Datasets included: {len(all_dataset_metrics)}")
    for sys_name, row in arch_agg["systems"].items():
        print(f"  {sys_name:<14} mean_score={row['mean_judge_score']}  "
              f"mean_cost={row['mean_cost_per_turn']}")

    return arch_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Research metrics post-processor.\n\n"
            "MODE 1 — Per-dataset (run after each dataset finishes):\n"
            "  python -m eval.research_metrics \\\n"
            "    --run-dir results/raw/benchmarks/qwen14b_arch_c/dataset_03_sql_databases\n\n"
            "  Place laaj.json in that folder first, then run to get all charts + metrics.\n\n"
            "MODE 2 — Aggregate (run after ALL datasets are done):\n"
            "  python -m eval.research_metrics \\\n"
            "    --aggregate results/raw/benchmarks/qwen14b_arch_c\n\n"
            "  Reads metrics_research.json from each sub-folder and writes arch_metrics_research.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        help="Dataset folder containing results.json (and optionally laaj.json). "
             "Outputs written into this same folder."
    )
    group.add_argument(
        "--aggregate",
        help="Arch folder containing dataset sub-folders with metrics_research.json files."
    )
    args = parser.parse_args()

    if args.run_dir:
        process_single_dataset(Path(args.run_dir))
    else:
        process_aggregate(Path(args.aggregate))


if __name__ == "__main__":
    main()
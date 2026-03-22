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

# Relative unit pricing (you can update per paper setup).
COST_PER_1M_TOKENS = {
    "main": 1.00,
    "curator": 0.50,
    "embedding": 0.10,
}


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
    pipeline = turn_log.get("pipeline_details") or {}
    context_tokens = pipeline.get("context_tokens_used", 0) or turn_log.get("context_tokens", 0) or 0

    # Main tokens proxy: context + completion approximation
    completion = _estimate_completion_tokens(turn_log.get("response", ""))
    main_tokens = max(0, int(context_tokens)) + completion

    curator_tokens = 0
    if system_name == "hiermem":
        strategy = (pipeline.get("curator_strategy") or "").upper()
        if strategy == "HYBRID":
            curator_tokens = max(32, int(main_tokens * 0.10))
        elif strategy:
            curator_tokens = max(16, int(main_tokens * 0.04))

    embedding_tokens = 0
    if system_name in ("rag", "rag_summary"):
        # Approximate retrieval embedding overhead per turn.
        queries = pipeline.get("semantic_queries") or []
        if queries:
            embedding_tokens = 120 * len(queries)
        else:
            embedding_tokens = 120

    return TurnCost(
        main_tokens=float(main_tokens),
        curator_tokens=float(curator_tokens),
        embedding_tokens=float(embedding_tokens),
    )


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
        turn = cp.get("turn")
        if turn is None:
            continue
        dedup[turn] = cp
    return [dedup[t] for t in sorted(dedup.keys())]


def _ensure_laaj_template(dataset_dir: Path, dataset_system_results: Dict[str, List[Dict[str, Any]]], judge_model: str) -> Path:
    laaj_path = dataset_dir / "laaj.json"
    if laaj_path.exists():
        return laaj_path

    judgments = []
    for system_name, convos in dataset_system_results.items():
        for conv in convos:
            cid = conv.get("conversation_id", "unknown")
            for cp in _collect_checkpoints(conv):
                judgments.append(
                    {
                        "system": system_name,
                        "conversation_id": cid,
                        "turn": cp.get("turn"),
                        "constraint_tested": cp.get("constraint_tested", ""),
                        "prompt": cp.get("test", ""),
                        "assistant_response": "",
                        "constraint_adherence": None,
                        "response_quality": None,
                        "conversational_coherence": None,
                        "weighted_score": None,
                        "reasoning": "",
                    }
                )

    template = {
        "metadata": {
            "judge_model": judge_model,
            "instructions": (
                "Fill each judgment with scores in [1,10]. "
                "weighted_score = 0.5*constraint_adherence + 0.3*response_quality + 0.2*conversational_coherence"
            ),
        },
        "judgments": judgments,
    }

    with laaj_path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    return laaj_path


def _load_laaj_scores(laaj_path: Path) -> Dict[Tuple[str, str, int], float]:
    if not laaj_path.exists():
        return {}
    with laaj_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    judgments = payload.get("judgments", []) if isinstance(payload, dict) else []
    scores: Dict[Tuple[str, str, int], float] = {}
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


def _load_external_judge_scores(judge_results_path: Optional[Path]) -> Dict[Tuple[str, str, int], float]:
    """Load judge scores from external evaluation outputs.

    Supports both schemas:
      1) metrics.json (legacy):
         {system: {judge_scores: {detail: [{convo, turn, judge_score}]}}}

      2) metrics_rescored_fixed.json (anonymous system_a/system_b format):
         {
           system_evaluations: {system_a: {checkpoints: [{turn, weighted_score}]}, ...},
           system_identity_reveal: {system_a: "hiermem", ...}
         }

    Returns score map keyed by (system_name, conversation_id_or_wildcard, turn).
    Uses conversation wildcard "*" when a source file does not carry convo ids.
    """
    if not judge_results_path or not judge_results_path.exists():
        return {}

    with judge_results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out: Dict[Tuple[str, str, int], float] = {}

    # Schema 1: metrics.json style
    if isinstance(payload, dict) and "system_evaluations" not in payload:
        for system_name, sys_block in payload.items():
            if not isinstance(sys_block, dict):
                continue
            detail = (sys_block.get("judge_scores") or {}).get("detail", [])
            for d in detail:
                convo = str(d.get("convo", "*"))
                turn = d.get("turn")
                score = d.get("judge_score")
                if turn is None or score is None:
                    continue
                out[(str(system_name), convo, int(turn))] = float(score)
        return out

    # Schema 2: metrics_rescored_fixed.json style
    system_evals = payload.get("system_evaluations", {}) if isinstance(payload, dict) else {}
    reveal = payload.get("system_identity_reveal", {}) if isinstance(payload, dict) else {}

    for anon_key, eval_block in system_evals.items():
        real_name = reveal.get(anon_key, anon_key)
        checkpoints = eval_block.get("checkpoints", []) if isinstance(eval_block, dict) else []
        for cp in checkpoints:
            turn = cp.get("turn")
            score = cp.get("weighted_score")
            if turn is None or score is None:
                continue
            # No convo id in this schema; use wildcard and resolve per dataset later.
            out[(str(real_name), "*", int(turn))] = float(score)

    return out


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

        for conv in convos:
            cid = conv.get("conversation_id", "unknown")
            turns = _iter_turn_logs(conv)
            budget = conv.get("config", {}).get("context_budget", 8192)

            for t in turns:
                lat = t.get("latency_seconds")
                if isinstance(lat, (int, float)):
                    turn_latencies.append(float(lat))

                tc = _estimate_turn_cost(system_name, t)
                # NOTE: all cost figures are ESTIMATED from token counts, not observed
                turn_costs.append(tc.total_cost_units)
                main_tokens.append(tc.main_tokens)
                curator_tokens.append(tc.curator_tokens)
                embedding_tokens.append(tc.embedding_tokens)

                pipeline = t.get("pipeline_details") or {}
                ctx_used = pipeline.get("context_tokens_used", 0) or t.get("context_tokens", 0) or 0
                if budget and ctx_used:
                    context_util.append(min(1.0, float(ctx_used) / float(budget)))

            for cp in _collect_checkpoints(conv):
                checkpoint_expected += 1
                cp_turn = int(cp.get("turn", 0))
                key = (system_name, cid, cp_turn)
                wildcard_key = (system_name, "*", cp_turn)
                score = None
                if key in laaj_scores:
                    score = laaj_scores[key]
                elif wildcard_key in laaj_scores:
                    score = laaj_scores[wildcard_key]
                elif cp_turn % 2 == 1:
                    # Some judge exports (e.g., metrics_rescored_fixed.json) use
                    # assistant-turn indexing (10,20,30,...) while results.json may
                    # store absolute odd turns (19,39,59,...). Normalize fallback.
                    normalized_turn = (cp_turn + 1) // 2
                    norm_key = (system_name, cid, normalized_turn)
                    norm_wildcard = (system_name, "*", normalized_turn)
                    if norm_key in laaj_scores:
                        score = laaj_scores[norm_key]
                    elif norm_wildcard in laaj_scores:
                        score = laaj_scores[norm_wildcard]

                if score is not None:
                    judge_scores.append(score)
                    degradation_by_turn.setdefault(cp_turn, []).append(score)
                    # Track first turn where a violation was judged
                    if score < VIOLATION_THRESHOLD and first_violation_turn is None:
                        first_violation_turn = cp_turn

        out["coverage"]["judge_scores_expected"] += checkpoint_expected
        out["coverage"]["judge_scores_present"] += len(judge_scores)

        # --- Aggregate scores — never default to 0 when absent ---
        avg_score: Optional[float] = mean(judge_scores) if judge_scores else None
        total_cost = float(sum(turn_costs))

        # cost_per_quality_point is undefined (None) when judge scores are missing.
        # Never substitute 0 or a sentinel like 999 — callers must handle None.
        cqp: Optional[float] = (total_cost / avg_score) if (avg_score and avg_score > 0) else None

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
            "total_cost_units": round(total_cost, 8),
            "avg_main_tokens": round(mean(main_tokens), 2) if main_tokens else None,
            "avg_curator_tokens": round(mean(curator_tokens), 2) if curator_tokens else None,
            "avg_embedding_tokens": round(mean(embedding_tokens), 2) if embedding_tokens else None,
            "curator_token_share": round(sum(curator_tokens) / total_tokens_all, 5),
            "context_utilization_pct": round(100.0 * mean(context_util), 2) if context_util else None,
            # --- Judge scores (OBSERVED, only present when laaj.json is filled) ---
            "judge_score_avg": round(avg_score, 4) if avg_score is not None else None,
            "judge_samples": len(judge_scores),
            # cost_per_quality_point is None when judge scores are absent — never 0 or fake sentinel
            "cost_per_quality_point": round(cqp, 8) if cqp is not None else None,
            # --- Degradation & survival (derived from LAAJ, None when no scores) ---
            "first_violation_turn": first_violation_turn,
            "constraint_survival_rate": survival_rate,
            "degradation_curve": degradation_curve,
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
        "total_cost_units",
        "cost_per_quality_point",
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

    # If every bar would be 0, skip this chart and let visualization_status.json
    # explain that token fields are missing for this dataset.
    if all((m + c + e) <= 0 for m, c, e in zip(main_vals, cur_vals, emb_vals)):
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


def _write_visualization_status(
    dataset_dir: Path,
    metrics_payload: Dict[str, Any],
    charts_written: List[str],
    charts_skipped: List[str],
) -> None:
    """Write visualization_status.json explaining which charts exist and why others were skipped."""
    skip_reasons: Dict[str, str] = {}
    if "cost_vs_quality.png" in charts_skipped:
        skip_reasons["cost_vs_quality.png"] = (
            "No systems in this dataset had both a judge_score_avg and "
            "avg_cost_per_turn_units. Fill laaj.json and re-run to generate."
        )
    if "latency_comparison.png" in charts_skipped:
        skip_reasons["latency_comparison.png"] = (
            "No avg_latency_seconds data found for any system in this dataset."
        )
    if "token_composition.png" in charts_skipped:
        skip_reasons["token_composition.png"] = (
            "Token fields were missing/zero for all systems, so composition would be empty."
        )

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
    judge_model: str,
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
    system_dataset_count: Dict[str, int] = {}

    for dataset_name, dmx in all_dataset_metrics.items():
        for sys_name, row in dmx.get("systems", {}).items():
            score = row.get("judge_score_avg")
            cost = row.get("avg_cost_per_turn_units")
            system_dataset_count[sys_name] = system_dataset_count.get(sys_name, 0) + 1
            if score is not None:
                system_scores.setdefault(sys_name, []).append(score)
            if cost is not None:
                system_costs.setdefault(sys_name, []).append(cost)

    agg_systems: Dict[str, Any] = {}
    all_sys = set(list(system_scores) + list(system_costs) + list(system_dataset_count))
    for sys_name in all_sys:
        scores = system_scores.get(sys_name, [])
        costs = system_costs.get(sys_name, [])
        agg_systems[sys_name] = {
            "datasets_included": system_dataset_count.get(sys_name, 0),
            "mean_judge_score": round(mean(scores), 4) if scores else None,
            # variance() requires ≥2 points; guard explicitly
            "score_variance": round(variance(scores), 6) if len(scores) > 1 else None,
            "mean_cost_per_turn": round(mean(costs), 8) if costs else None,
            "cost_variance": round(variance(costs), 10) if len(costs) > 1 else None,
        }

    return {
        "metadata": {
            "arch": arch_name,
            "source_run_dir": source_run_dir,
            "judge_model": judge_model,
            "datasets_count": len(all_dataset_metrics),
            "cost_units": COST_PER_1M_TOKENS,
        },
        "systems": agg_systems,
        # Verbatim per-dataset breakdown nested here for one-stop reading
        "datasets": all_dataset_metrics,
    }


def process_run(run_dir: Path, arch: Optional[str], judge_model: str, judge_results_path: Optional[Path]) -> Path:
    """Main entry point. Reads results.json, writes dataset folders and arch JSON."""
    results = _load_results(run_dir)
    grouped = _group_by_dataset(results)

    arch_name = arch or run_dir.name
    arch_root = run_dir.parent / arch_name
    arch_root.mkdir(parents=True, exist_ok=True)

    all_dataset_metrics: Dict[str, Dict[str, Any]] = {}

    for dataset_name, dataset_system_results in sorted(grouped.items()):
        dataset_dir = arch_root / _safe_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        _write_dataset_payloads(dataset_dir, dataset_system_results)
        laaj_path = _ensure_laaj_template(dataset_dir, dataset_system_results, judge_model)
        laaj_scores = _load_laaj_scores(laaj_path)
        external_scores = _load_external_judge_scores(judge_results_path)
        # Local laaj.json entries should override external imports for iterative curation.
        merged_scores = dict(external_scores)
        merged_scores.update(laaj_scores)

        metrics_payload = _build_metrics_for_dataset(dataset_name, dataset_system_results, merged_scores)

        with (dataset_dir / "metrics_research.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        _write_summary_csv(dataset_dir, metrics_payload)

        # Track which charts were written vs skipped so we can report honestly
        charts_written: List[str] = []
        charts_skipped: List[str] = []

        def _record(name: str, wrote: bool) -> None:
            (charts_written if wrote else charts_skipped).append(name)

        _record("cost_vs_quality.png",    _plot_quality_cost(dataset_dir, metrics_payload))
        _record("latency_comparison.png", _plot_latency(dataset_dir, metrics_payload))
        _record("token_composition.png",  _plot_token_breakdown(dataset_dir, metrics_payload))

        _write_visualization_status(dataset_dir, metrics_payload, charts_written, charts_skipped)

        all_dataset_metrics[dataset_name] = metrics_payload

        status = metrics_payload.get("dataset_status", "unknown")
        written_str = ", ".join(charts_written) or "none"
        skipped_str = ", ".join(charts_skipped) or "none"
        print(
            f"  [{status}] {dataset_name}: "
            f"charts_written=[{written_str}]  skipped=[{skipped_str}]"
        )

    # Write arch-level aggregation
    arch_agg = _build_arch_aggregation(
        arch_name=arch_name,
        source_run_dir=str(run_dir),
        judge_model=judge_model,
        all_dataset_metrics=all_dataset_metrics,
    )
    with (arch_root / "arch_metrics_research.json").open("w", encoding="utf-8") as f:
        json.dump(arch_agg, f, ensure_ascii=False, indent=2)

    return arch_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified research metrics post-processor")
    parser.add_argument("--run-dir", required=True, help="Path to run directory containing results.json")
    parser.add_argument("--arch", default=None, help="Architecture output folder name (e.g., qwen14b_arch_b)")
    parser.add_argument("--judge-model", default="gpt-4.1", help="Judge model label stored in laaj metadata")
    parser.add_argument(
        "--judge-results",
        default=None,
        help="Optional path to external judge results (metrics.json or metrics_rescored_fixed.json)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    judge_results_path = Path(args.judge_results) if args.judge_results else None
    arch_root = process_run(
        run_dir=run_dir,
        arch=args.arch,
        judge_model=args.judge_model,
        judge_results_path=judge_results_path,
    )
    print(f"Unified research outputs generated at: {arch_root}")


if __name__ == "__main__":
    main()
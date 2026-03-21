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
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
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
        turn_latencies = []
        turn_costs = []
        main_tokens = []
        curator_tokens = []
        embedding_tokens = []
        context_util = []
        judge_scores = []
        checkpoint_expected = 0

        for conv in convos:
            cid = conv.get("conversation_id", "unknown")
            turns = _iter_turn_logs(conv)
            budget = conv.get("config", {}).get("context_budget", 8192)

            for t in turns:
                lat = t.get("latency_seconds")
                if isinstance(lat, (int, float)):
                    turn_latencies.append(float(lat))

                tc = _estimate_turn_cost(system_name, t)
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
                key = (system_name, cid, int(cp.get("turn", 0)))
                if key in laaj_scores:
                    judge_scores.append(laaj_scores[key])

        out["coverage"]["judge_scores_expected"] += checkpoint_expected
        out["coverage"]["judge_scores_present"] += len(judge_scores)

        avg_score = mean(judge_scores) if judge_scores else None
        total_cost = float(sum(turn_costs))
        cqp = (total_cost / avg_score) if (avg_score and avg_score > 0) else None

        out["systems"][system_name] = {
            "num_conversations": len(convos),
            "avg_latency_seconds": round(mean(turn_latencies), 3) if turn_latencies else None,
            "avg_cost_per_turn_units": round(mean(turn_costs), 8) if turn_costs else None,
            "total_cost_units": round(total_cost, 8),
            "avg_main_tokens": round(mean(main_tokens), 2) if main_tokens else 0,
            "avg_curator_tokens": round(mean(curator_tokens), 2) if curator_tokens else 0,
            "avg_embedding_tokens": round(mean(embedding_tokens), 2) if embedding_tokens else 0,
            "curator_token_share": round(
                (sum(curator_tokens) / max(1.0, (sum(main_tokens) + sum(curator_tokens) + sum(embedding_tokens)))),
                5,
            ),
            "context_utilization_pct": round(100.0 * mean(context_util), 2) if context_util else None,
            "judge_score_avg": round(avg_score, 4) if avg_score is not None else None,
            "judge_samples": len(judge_scores),
            "cost_per_quality_point": round(cqp, 8) if cqp is not None else None,
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
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for system_name in SYSTEM_ORDER:
            if system_name not in metrics_payload["systems"]:
                continue
            row = {"system": system_name}
            row.update(metrics_payload["systems"][system_name])
            writer.writerow(row)


def _plot_quality_cost(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted = 0
    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        x = row.get("avg_cost_per_turn_units")
        y = row.get("judge_score_avg")
        if x is None or y is None:
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
        )
        ax.annotate(SYSTEM_LABELS.get(system_name, system_name), (x, y), xytext=(6, 6), textcoords="offset points")

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Average Cost per Turn (relative units)")
    ax.set_ylabel("LAAJ Score (1-10)")
    ax.set_title("Cost vs LAAJ Quality")
    ax.grid(alpha=0.25)
    ax.set_ylim(0.0, 10.5)
    fig.tight_layout()
    fig.savefig(dataset_dir / "cost_vs_quality.png", dpi=220)
    plt.close(fig)


def _plot_latency(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> None:
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
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylabel("Avg Latency per Turn (s)")
    ax.set_title("Latency Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(dataset_dir / "latency_comparison.png", dpi=220)
    plt.close(fig)


def _plot_token_breakdown(dataset_dir: Path, metrics_payload: Dict[str, Any]) -> None:
    systems = []
    main_vals = []
    cur_vals = []
    emb_vals = []

    for system_name in SYSTEM_ORDER:
        row = metrics_payload["systems"].get(system_name)
        if not row:
            continue
        systems.append(SYSTEM_LABELS.get(system_name, system_name))
        main_vals.append(row.get("avg_main_tokens", 0.0) or 0.0)
        cur_vals.append(row.get("avg_curator_tokens", 0.0) or 0.0)
        emb_vals.append(row.get("avg_embedding_tokens", 0.0) or 0.0)

    if not systems:
        return

    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, main_vals, label="Main")
    ax.bar(x, cur_vals, bottom=main_vals, label="Curator")
    ax.bar(x, emb_vals, bottom=np.array(main_vals) + np.array(cur_vals), label="Embedding")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20)
    ax.set_ylabel("Avg Tokens per Turn")
    ax.set_title("Token Composition by System")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(dataset_dir / "token_composition.png", dpi=220)
    plt.close(fig)


def process_run(run_dir: Path, arch: Optional[str], judge_model: str) -> Path:
    results = _load_results(run_dir)
    grouped = _group_by_dataset(results)

    arch_name = arch or run_dir.name
    arch_root = run_dir.parent / arch_name
    arch_root.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "metadata": {
            "arch": arch_name,
            "source_run_dir": str(run_dir),
            "judge_model": judge_model,
            "cost_units": COST_PER_1M_TOKENS,
        },
        "datasets": {},
    }

    for dataset_name, dataset_system_results in sorted(grouped.items()):
        dataset_dir = arch_root / _safe_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        _write_dataset_payloads(dataset_dir, dataset_system_results)
        laaj_path = _ensure_laaj_template(dataset_dir, dataset_system_results, judge_model)
        laaj_scores = _load_laaj_scores(laaj_path)

        metrics_payload = _build_metrics_for_dataset(dataset_name, dataset_system_results, laaj_scores)

        with (dataset_dir / "metrics_research.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        _write_summary_csv(dataset_dir, metrics_payload)
        _plot_quality_cost(dataset_dir, metrics_payload)
        _plot_latency(dataset_dir, metrics_payload)
        _plot_token_breakdown(dataset_dir, metrics_payload)

        all_metrics["datasets"][dataset_name] = metrics_payload

    with (arch_root / "arch_metrics_research.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    return arch_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified research metrics post-processor")
    parser.add_argument("--run-dir", required=True, help="Path to run directory containing results.json")
    parser.add_argument("--arch", default=None, help="Architecture output folder name (e.g., qwen14b_arch_b)")
    parser.add_argument("--judge-model", default="gpt-4.1", help="Judge model label stored in laaj metadata")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    arch_root = process_run(run_dir=run_dir, arch=args.arch, judge_model=args.judge_model)
    print(f"Unified research outputs generated at: {arch_root}")


if __name__ == "__main__":
    main()

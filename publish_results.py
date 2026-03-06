"""
Publish best results — Extract key metrics from a run directory and save
a clean summary to results/published/.

Usage:
    python publish_results.py results/raw/benchmarks/qwen14b_run
    python publish_results.py results/raw/benchmarks/gemma12b_run --benchmark constraint_tracking

Reads: <run_dir>/metrics_rescored.json (preferred) or metrics.json
Reads: <run_dir>/run_info.json for model/provider metadata

Writes:
  results/published/<benchmark>/<model_slug>.json  — best result per model
  results/published/leaderboard.json               — aggregated table
"""
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime


PUBLISHED_DIR = Path("results/published")


def _load_run(run_dir: Path):
    """Load metrics and run_info from a run directory."""
    metrics_file = run_dir / "metrics_rescored.json"
    if not metrics_file.exists():
        metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics file found in {run_dir}")

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    run_info_file = run_dir / "run_info.json"
    run_info = json.loads(run_info_file.read_text(encoding="utf-8")) if run_info_file.exists() else {}

    return metrics, run_info


def _extract_system_summary(system_metrics: dict) -> dict:
    """Extract the key numbers for one system from its metrics block.
    
    Returns aggregate numbers plus a per-conversation breakdown so results
    from different test datasets (cooking, software, legal...) stay visible.
    """
    js = system_metrics.get("judge_scores", {})
    pts = system_metrics.get("per_turn_scores", {})

    # Group judge scores by conversation
    by_convo: dict = {}
    for d in js.get("detail", []):
        cid = d["convo"]
        if cid not in by_convo:
            by_convo[cid] = {"scores": [], "turns": {}}
        by_convo[cid]["scores"].append(d["judge_score"])
        by_convo[cid]["turns"][str(d["turn"])] = d["judge_score"]

    by_conversation = {
        cid: {
            "judge_score_avg": round(sum(v["scores"]) / len(v["scores"]), 2),
            "judge_score_by_turn": v["turns"],
        }
        for cid, v in by_convo.items()
    }

    return {
        "judge_score_avg": js.get("avg_score"),            # PRIMARY: 1-10
        "keyword_score_avg": pts.get("avg_score"),         # 0-100
        "task_accuracy": round(system_metrics.get("task_accuracy", 0) * 100, 1),  # %
        "cvr": round(system_metrics.get("constraint_violation_rate", 0) * 100, 1),  # %
        "total_turns": system_metrics.get("total_turns_processed", 0),
        "by_conversation": by_conversation,                # per-dataset breakdown
    }


def build_summary(run_dir: Path, benchmark: str) -> dict:
    """Build a clean publishable summary from a run directory."""
    metrics, run_info = _load_run(run_dir)

    models = run_info.get("models", {})
    provider = run_info.get("provider", "unknown")
    main_model = models.get("main", "unknown").split("/")[-1]  # strip "ollama/"

    systems = {}
    for sys_name, sys_metrics in metrics.items():
        if sys_name == "pairwise_comparison":
            continue
        systems[sys_name] = _extract_system_summary(sys_metrics)

    pairwise = metrics.get("pairwise_comparison", {})
    pairwise_summary = None
    if pairwise and "system_a" in pairwise:
        pairwise_summary = {
            "system_a": pairwise["system_a"],
            "system_b": pairwise["system_b"],
            "wins_a": pairwise["wins_a"],
            "wins_b": pairwise["wins_b"],
            "ties": pairwise["ties"],
            "total": pairwise["total"],
            "win_rate_a": pairwise["win_rate_a"],
            "win_rate_b": pairwise["win_rate_b"],
        }

    return {
        "benchmark": benchmark,
        "model": main_model,
        "provider": provider,
        "curator_model": models.get("curator", ""),
        "published_at": datetime.now().strftime("%Y-%m-%d"),
        "run_dir": str(run_dir),
        "num_conversations": run_info.get("num_conversations", "?"),
        "systems": systems,
        "pairwise": pairwise_summary,
    }


def publish(run_dir: Path, benchmark: str, force: bool = False):
    """Publish a run summary. Skips if existing result is already better."""
    summary = build_summary(run_dir, benchmark)

    model_slug = summary["model"].replace(":", "-")
    provider = summary["provider"]

    pub_dir = PUBLISHED_DIR / benchmark
    pub_dir.mkdir(parents=True, exist_ok=True)

    out_path = pub_dir / f"{provider}_{model_slug}.json"

    # Get new hiermem judge score
    new_score = summary["systems"].get("hiermem", {}).get("judge_score_avg", 0) or 0

    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        old_score = existing.get("systems", {}).get("hiermem", {}).get("judge_score_avg", 0) or 0
        if old_score >= new_score:
            print(f"  Existing result ({old_score}/10) >= new ({new_score}/10), skipping.")
            print(f"  Use --force to overwrite.")
            return

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Published: {out_path}")

    # Print human-readable summary
    print(f"\n  Model: {summary['model']} ({provider})")
    print(f"  Benchmark: {benchmark}  |  Conversations: {summary['num_conversations']}")
    print(f"\n  {'System':<14} {'Judge':>9}  {'KW Score':>10}  {'Accuracy':>10}  {'CVR':>6}")
    print(f"  {'-'*55}")
    for sys_name, s in summary["systems"].items():
        j = f"{s['judge_score_avg']}/10" if s['judge_score_avg'] is not None else "n/a"
        kw = f"{s['keyword_score_avg']}/100" if s['keyword_score_avg'] is not None else "n/a"
        print(f"  {sys_name:<14} {j:>9}  {kw:>10}  {s['task_accuracy']:>9}%  {s['cvr']:>5}%")
        for cid, cv in s.get("by_conversation", {}).items():
            print(f"    {cid:<50} {cv['judge_score_avg']}/10")

    if summary["pairwise"]:
        p = summary["pairwise"]
        print(f"\n  Pairwise: {p['system_a']} {p['wins_a']}W  "
              f"{p['system_b']} {p['wins_b']}W  ties={p['ties']}")

    # Update leaderboard
    _update_leaderboard(summary)


def _update_leaderboard(summary: dict):
    """Update the aggregated leaderboard JSON."""
    lb_path = PUBLISHED_DIR / "leaderboard.json"
    leaderboard = json.loads(lb_path.read_text(encoding="utf-8")) if lb_path.exists() else []

    # Remove existing entry for same model+benchmark
    key = (summary["model"], summary["benchmark"])
    leaderboard = [e for e in leaderboard
                   if not (e["model"] == key[0] and e["benchmark"] == key[1])]

    hiermem = summary["systems"].get("hiermem", {})
    raw_llm = summary["systems"].get("raw_llm", {})
    pairwise = summary.get("pairwise", {})

    leaderboard.append({
        "model": summary["model"],
        "provider": summary["provider"],
        "benchmark": summary["benchmark"],
        "published_at": summary["published_at"],
        "hiermem_judge": hiermem.get("judge_score_avg"),
        "raw_llm_judge": raw_llm.get("judge_score_avg"),
        "hiermem_accuracy": hiermem.get("task_accuracy"),
        "raw_llm_accuracy": raw_llm.get("task_accuracy"),
        "pairwise_hiermem_wins": pairwise.get("wins_a") if pairwise else None,
        "pairwise_total": pairwise.get("total") if pairwise else None,
        "num_conversations": summary["num_conversations"],
    })

    # Sort: best hiermem judge score first
    leaderboard.sort(key=lambda e: e.get("hiermem_judge") or 0, reverse=True)
    lb_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    print(f"\n  Leaderboard updated: {lb_path}  ({len(leaderboard)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish benchmark results")
    parser.add_argument("run_dir", help="Run directory (e.g. results/raw/benchmarks/qwen14b_run)")
    parser.add_argument("--benchmark", default="constraint_tracking",
                        help="Benchmark name (default: constraint_tracking)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite even if existing result is better")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} not found")
        sys.exit(1)

    publish(run_dir, args.benchmark, force=args.force)


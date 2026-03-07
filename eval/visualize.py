"""
Visualization — Generate comparison charts for evaluation results.
"""

import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


SYSTEM_COLORS = {
    "hiermem":    "#2ecc71",   # green — our system
    "raw_llm":    "#e74c3c",   # red
    "rag":        "#3498db",   # blue
    "rag_summary":"#f39c12",  # orange
    "memgpt_style":"#9b59b6", # purple
}
SYSTEM_LABELS = {
    "hiermem":    "HierMem (ours)",
    "raw_llm":    "Raw LLM",
    "rag":        "RAG",
    "rag_summary":"RAG + Summary",
    "memgpt_style":"MemGPT-style",
}


def _color(system_name: str) -> str:
    return SYSTEM_COLORS.get(system_name, "#95a5a6")


def _label(system_name: str) -> str:
    return SYSTEM_LABELS.get(system_name, system_name)


def plot_degradation_curves(metrics: dict, output_path: Path):
    """Plot accuracy vs conversation length for all systems."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for system_name, system_metrics in metrics.items():
        if system_name == "pairwise_comparison":
            continue
        curve = system_metrics.get("degradation_curve", {})
        if curve:
            turns = sorted(int(k) for k in curve.keys())
            accuracies = [curve[str(t)] if str(t) in curve else curve.get(t, 0)
                         for t in turns]
            lw = 2.5 if system_name == "hiermem" else 1.8
            ax.plot(turns, [a * 100 for a in accuracies],
                    marker='o', linewidth=lw, label=_label(system_name),
                    color=_color(system_name))

    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Constraint Compliance: Accuracy vs Conversation Length", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 108)
    fig.tight_layout()
    fig.savefig(output_path / "degradation_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path / 'degradation_curves.png'}")


def plot_judge_score_trend(metrics: dict, output_path: Path):
    """Line chart: judge score (1-10) at each checkpoint turn — PRIMARY paper figure."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for system_name, system_metrics in metrics.items():
        if system_name == "pairwise_comparison":
            continue
        summary = system_metrics.get("judge_scores", {}).get("summary", {})
        if not summary:
            continue
        turns = sorted(int(k) for k in summary.keys())
        scores = [summary[str(t)] for t in turns]
        lw = 2.5 if system_name == "hiermem" else 1.8
        ax.plot(turns, scores, marker='o', linewidth=lw,
                label=_label(system_name), color=_color(system_name))

    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Judge Score (1–10)", fontsize=12)
    ax.set_title("Constraint Following Quality: LLM Judge Score Over Turns", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    fig.tight_layout()
    fig.savefig(output_path / "judge_score_trend.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path / 'judge_score_trend.png'}")


def plot_metric_comparison(metrics: dict, output_path: Path):
    """Bar chart comparing key metrics across systems."""
    if not HAS_MATPLOTLIB:
        return

    systems = [s for s in metrics.keys() if s != "pairwise_comparison"]
    colors = [_color(s) for s in systems]
    labels = [_label(s) for s in systems]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Judge score (primary)
    scores = [metrics[s].get("judge_scores", {}).get("avg_score", 0) for s in systems]
    bars = axes[0].bar(labels, scores, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title("Avg Judge Score (1–10)\n← Primary Metric", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 11)
    for bar, val in zip(bars, scores):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=25)

    # Task Accuracy
    accuracies = [metrics[s].get("task_accuracy", 0) * 100 for s in systems]
    bars = axes[1].bar(labels, accuracies, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_title("Checkpoint Accuracy (%)", fontsize=11)
    axes[1].set_ylim(0, 108)
    for bar, val in zip(bars, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
    axes[1].tick_params(axis='x', rotation=25)

    # CVR
    cvrs = [metrics[s].get("constraint_violation_rate", 0) * 100 for s in systems]
    bars = axes[2].bar(labels, cvrs, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_title("Constraint Violation Rate (%)\n← Lower is Better", fontsize=11)
    axes[2].set_ylim(0, 70)
    for bar, val in zip(bars, cvrs):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
    axes[2].tick_params(axis='x', rotation=25)

    fig.suptitle("System Comparison: Constraint Tracking Benchmark", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "metric_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path / 'metric_comparison.png'}")


def plot_context_tokens_per_turn(results: dict, output_path: Path):
    """Line chart: context tokens used per turn — shows O(1) vs linear growth."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for system_name, system_results in results.items():
        turn_tokens = {}  # turn -> list of token counts (avg across convos)
        for conv in system_results:
            for t in conv.get("turn_logs", conv.get("results", [])):
                turn = t.get("turn", 0)
                pd = t.get("pipeline_details") or {}
                ctx = pd.get("context_tokens_used", 0) or t.get("context_tokens", 0)
                if ctx > 0:
                    turn_tokens.setdefault(turn, []).append(ctx)

        if not turn_tokens:
            continue
        turns = sorted(turn_tokens.keys())
        avgs = [sum(turn_tokens[t]) / len(turn_tokens[t]) for t in turns]
        lw = 2.5 if system_name == "hiermem" else 1.8
        ax.plot(turns, avgs, linewidth=lw, label=_label(system_name), color=_color(system_name))

    ax.axhline(y=6000, color='black', linestyle='--', linewidth=1, alpha=0.4, label='Budget (6000 tok)')
    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Context Tokens Used", fontsize=12)
    ax.set_title("Context Token Usage Per Turn", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 7000)
    fig.tight_layout()
    fig.savefig(output_path / "context_tokens_per_turn.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path / 'context_tokens_per_turn.png'}")


def plot_latency_per_turn(results: dict, output_path: Path):
    """Line chart: latency per turn across systems."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for system_name, system_results in results.items():
        turn_latency = {}
        for conv in system_results:
            for t in conv.get("turn_logs", conv.get("results", [])):
                turn = t.get("turn", 0)
                lat = t.get("latency_seconds", 0)
                if lat > 0:
                    turn_latency.setdefault(turn, []).append(lat)

        if not turn_latency:
            continue
        turns = sorted(turn_latency.keys())
        avgs = [sum(turn_latency[t]) / len(turn_latency[t]) for t in turns]
        lw = 2.5 if system_name == "hiermem" else 1.8
        ax.plot(turns, avgs, linewidth=lw, label=_label(system_name), color=_color(system_name))

    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("Avg Latency (seconds)", fontsize=12)
    ax.set_title("Per-Turn Latency (local Ollama qwen2.5:14b)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "latency_per_turn.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path / 'latency_per_turn.png'}")


def plot_llm_calls_per_turn(results: dict, output_path: Path):
    """Stacked/grouped bar: LLM calls per turn for HierMem (passthrough vs HYBRID)."""
    if not HAS_MATPLOTLIB:
        return

    hiermem_results = results.get("hiermem")
    if not hiermem_results:
        return

    turn_modes = {}
    for conv in hiermem_results:
        for t in conv.get("turn_logs", conv.get("results", [])):
            turn = t.get("turn", 0)
            pd = t.get("pipeline_details") or {}
            warnings = pd.get("warnings", [])
            sources = pd.get("sources_used", [])
            is_pass = (any("passthrough" in str(w) for w in warnings) or
                       any("passthrough" in str(s) for s in sources))
            turn_modes.setdefault(turn, []).append(1 if is_pass else 2)

    if not turn_modes:
        return

    turns = sorted(turn_modes.keys())
    avg_calls = [sum(turn_modes[t]) / len(turn_modes[t]) for t in turns]

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_colors = ["#2ecc71" if c <= 1.1 else "#f39c12" for c in avg_calls]
    ax.bar(turns, avg_calls, color=bar_colors, edgecolor='white', linewidth=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='raw_llm/rag baseline (1 call)')
    ax.set_xlabel("Conversation Turn", fontsize=11)
    ax.set_ylabel("LLM Calls per Turn", fontsize=11)
    ax.set_title("HierMem: LLM Calls per Turn\n(green=passthrough/1 call, orange=HYBRID/2 calls)", fontsize=11)
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "hiermem_llm_calls.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path / 'hiermem_llm_calls.png'}")


def generate_all_charts(results_dir: Path):
    """Generate all visualization charts from results directory."""
    metrics_file = results_dir / "metrics_rescored.json"
    if not metrics_file.exists():
        metrics_file = results_dir / "metrics.json"
    results_file = results_dir / "results.json"

    if not metrics_file.exists():
        print(f"No metrics JSON found in {results_dir}")
        return

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else {}

    plot_degradation_curves(metrics, results_dir)
    plot_judge_score_trend(metrics, results_dir)
    plot_metric_comparison(metrics, results_dir)
    if results:
        plot_context_tokens_per_turn(results, results_dir)
        plot_latency_per_turn(results, results_dir)
        plot_llm_calls_per_turn(results, results_dir)

    print(f"\nAll charts saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results directory path")
    args = parser.parse_args()
    generate_all_charts(Path(args.results))


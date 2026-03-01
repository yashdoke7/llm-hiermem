"""
Visualization — Generate comparison charts for evaluation results.
"""

import json
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_degradation_curves(metrics: dict, output_path: Path):
    """Plot accuracy vs conversation length for all systems."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping visualization")
        return
    
    plt.figure(figsize=(10, 6))
    
    for system_name, system_metrics in metrics.items():
        curve = system_metrics.get("degradation_curve", {})
        if curve:
            turns = sorted(int(k) for k in curve.keys())
            accuracies = [curve[str(t)] if str(t) in curve else curve.get(t, 0) 
                         for t in turns]
            plt.plot(turns, [a * 100 for a in accuracies], 
                    marker='o', linewidth=2, label=system_name)
    
    plt.xlabel("Conversation Turn", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Degradation Curve: Accuracy vs Conversation Length", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_path / "degradation_curves.png", dpi=150)
    plt.close()
    print(f"Saved degradation curve to {output_path / 'degradation_curves.png'}")


def plot_metric_comparison(metrics: dict, output_path: Path):
    """Bar chart comparing key metrics across systems."""
    if not HAS_MATPLOTLIB:
        return
    
    systems = list(metrics.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Task Accuracy
    accuracies = [metrics[s].get("task_accuracy", 0) * 100 for s in systems]
    axes[0].bar(systems, accuracies, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    axes[0].set_title("Task Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].tick_params(axis='x', rotation=30)
    
    # Constraint Violation Rate
    cvrs = [metrics[s].get("constraint_violation_rate", 0) * 100 for s in systems]
    axes[1].bar(systems, cvrs, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    axes[1].set_title("Constraint Violation Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis='x', rotation=30)
    
    # Total turns processed
    turns = [metrics[s].get("total_turns_processed", 0) for s in systems]
    axes[2].bar(systems, turns, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    axes[2].set_title("Total Turns Processed")
    axes[2].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(output_path / "metric_comparison.png", dpi=150)
    plt.close()
    print(f"Saved metric comparison to {output_path / 'metric_comparison.png'}")


def generate_all_charts(results_dir: Path):
    """Generate all visualization charts from results directory."""
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"No metrics.json found in {results_dir}")
        return
    
    metrics = json.loads(metrics_file.read_text())
    plot_degradation_curves(metrics, results_dir)
    plot_metric_comparison(metrics, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results directory path")
    args = parser.parse_args()
    generate_all_charts(Path(args.results))

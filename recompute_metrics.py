"""Quick script to recompute metrics from existing results."""
import json
from pathlib import Path
from eval.metrics import compute_all_metrics

results_dir = Path("results/raw/benchmarks/constraint_tracking_ollama_llama3.1-8b_20260303_125139")
results = json.loads((results_dir / "results.json").read_text())
metrics = compute_all_metrics(results)
(results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

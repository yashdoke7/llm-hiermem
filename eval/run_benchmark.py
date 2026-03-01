"""
Benchmark Runner — Run all systems on all benchmarks.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from tqdm import tqdm

from core.pipeline import HierMemPipeline
from baselines.raw_llm import RawLLMBaseline
from baselines.rag_baseline import RAGBaseline
from baselines.rag_summary import RAGSummaryBaseline
from baselines.memgpt_style import MemGPTStyleBaseline
from eval.metrics import compute_all_metrics
import config


SYSTEMS = {
    "raw_llm": RawLLMBaseline,
    "rag": RAGBaseline,
    "rag_summary": RAGSummaryBaseline,
    "memgpt_style": MemGPTStyleBaseline,
    "hiermem": HierMemPipeline,
}


def load_benchmark(benchmark_name: str) -> List[Dict]:
    """Load a benchmark dataset."""
    datasets_dir = Path(__file__).parent / "datasets"
    
    # Look for JSON files in the benchmark directory
    benchmark_dir = datasets_dir / benchmark_name
    if benchmark_dir.is_dir():
        files = list(benchmark_dir.glob("*.json"))
        if files:
            all_data = []
            for f in files:
                all_data.extend(json.loads(f.read_text()))
            return all_data
    
    # Try direct file
    direct = datasets_dir / f"{benchmark_name}.json"
    if direct.exists():
        return json.loads(direct.read_text())
    
    raise FileNotFoundError(f"Benchmark '{benchmark_name}' not found in {datasets_dir}")


def run_system_on_conversation(system, conversation: Dict) -> Dict:
    """Run a single system on a single conversation and record results."""
    turns = conversation.get("turns", [])
    checkpoints = conversation.get("checkpoints", [])
    results = []
    
    for turn_data in turns:
        if turn_data.get("role") == "user":
            user_msg = turn_data["text"]
            result = system.process_turn(user_msg)
            results.append({
                "turn": turn_data.get("turn", 0),
                "user": user_msg,
                "response": result.assistant_response if hasattr(result, 'assistant_response') else str(result),
            })
    
    return {
        "conversation_id": conversation.get("conversation_id", "unknown"),
        "turns_processed": len(results),
        "results": results,
        "checkpoints": checkpoints,
    }


def run_benchmark(systems_to_run: List[str], benchmark_name: str,
                  output_dir: Path = None):
    """Run specified systems on a benchmark."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or config.RESULTS_PATH / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conversations = load_benchmark(benchmark_name)
    print(f"Loaded {len(conversations)} conversations from '{benchmark_name}'")
    
    all_results = {}
    
    for system_name in systems_to_run:
        if system_name not in SYSTEMS:
            print(f"Unknown system: {system_name}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running: {system_name}")
        print(f"{'='*60}")
        
        system_results = []
        
        for conv in tqdm(conversations, desc=system_name):
            # Create fresh system instance for each conversation
            if system_name == "hiermem":
                system = HierMemPipeline.create()
            else:
                system = SYSTEMS[system_name]()
            
            result = run_system_on_conversation(system, conv)
            system_results.append(result)
        
        all_results[system_name] = system_results
    
    # Save results
    results_file = output_dir / "results.json"
    results_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_file}")
    
    # Compute metrics
    metrics = compute_all_metrics(all_results)
    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_file}")
    
    return all_results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HierMem benchmarks")
    parser.add_argument("--systems", nargs="+", default=["all"],
                        help="Systems to evaluate (or 'all')")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark name (directory in eval/datasets/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()
    
    systems = list(SYSTEMS.keys()) if "all" in args.systems else args.systems
    output_dir = Path(args.output) if args.output else None
    
    run_benchmark(systems, args.benchmark, output_dir)

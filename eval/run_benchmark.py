"""
Benchmark Runner — Run all systems on all benchmarks with detailed logging.
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
from eval.metrics import compute_all_metrics, init_llm_judge
import config


SYSTEMS = {
    "raw_llm": RawLLMBaseline,
    "rag": RAGBaseline,
    "rag_summary": RAGSummaryBaseline,
    "memgpt_style": MemGPTStyleBaseline,
    "hiermem": HierMemPipeline,
}


def load_benchmark(benchmark_name: str) -> List[Dict]:
    """Load a benchmark dataset.
    
    Recursively discovers all *.json files under the benchmark directory,
    skipping metadata files (dataset_meta.json).  This supports the
    per-domain file layout (synthetic/, llm_generated/, real_exports/).
    """
    datasets_dir = Path(__file__).parent / "datasets"
    
    benchmark_dir = datasets_dir / benchmark_name
    if benchmark_dir.is_dir():
        files = sorted(benchmark_dir.rglob("*.json"))
        # Skip metadata files and raw export files
        skip_dirs = {"real_exports"}
        files = [f for f in files 
                 if f.name != "dataset_meta.json"
                 and not any(sd in f.parts for sd in skip_dirs)]
        if files:
            all_data = []
            for f in files:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    # Only include conversations with required benchmark fields
                    valid = [d for d in data if isinstance(d, dict) and "turns" in d and "checkpoints" in d]
                    all_data.extend(valid)
                elif isinstance(data, dict) and "turns" in data and "checkpoints" in data:
                    all_data.append(data)
            return all_data
    
    direct = datasets_dir / f"{benchmark_name}.json"
    if direct.exists():
        return json.loads(direct.read_text(encoding="utf-8"))
    
    raise FileNotFoundError(f"Benchmark '{benchmark_name}' not found in {datasets_dir}")


def run_system_on_conversation(system, system_name: str, conversation: Dict) -> Dict:
    """Run a single system on a single conversation with detailed per-step logging."""
    turns = conversation.get("turns", [])
    checkpoints = conversation.get("checkpoints", [])
    turn_logs = []
    total_start = time.time()
    
    for turn_data in turns:
        if turn_data.get("role") != "user":
            continue
        
        user_msg = turn_data["text"]
        turn_num = turn_data.get("turn", 0)
        turn_type = turn_data.get("type", "unknown")
        
        turn_start = time.time()
        
        try:
            result = system.process_turn(user_msg)
            response = result.assistant_response if hasattr(result, 'assistant_response') else str(result)
            
            # Build detailed turn log
            turn_log = {
                "turn": turn_num,
                "type": turn_type,
                "user": user_msg,
                "response": response,
                "latency_seconds": round(time.time() - turn_start, 2),
            }
            
            # HierMem-specific details
            if system_name == "hiermem" and hasattr(result, 'curator_decision'):
                cd = result.curator_decision
                turn_log["pipeline_details"] = {
                    "curator_strategy": cd.retrieval_strategy if cd else None,
                    "segments_fetched": cd.segments_to_fetch if cd else [],
                    "semantic_queries": cd.semantic_queries if cd else [],
                    "curator_reasoning": cd.reasoning if cd else "",
                    "context_tokens_used": result.tokens_used,
                    "sources_used": result.assembled_context.sources_used if result.assembled_context else [],
                    "zone_breakdown": result.assembled_context.zone_breakdown if result.assembled_context else {},
                    "warnings": result.warnings,
                }
            
            # Raw LLM specific
            if hasattr(result, 'context_tokens'):
                turn_log["context_tokens"] = result.context_tokens
            
        except Exception as e:
            response = f"ERROR: {e}"
            turn_log = {
                "turn": turn_num,
                "type": turn_type,
                "user": user_msg,
                "response": response,
                "latency_seconds": round(time.time() - turn_start, 2),
                "error": str(e),
            }
        
        turn_logs.append(turn_log)
    
    total_elapsed = round(time.time() - total_start, 2)
    
    # Capture final system state
    system_state = {}
    if system_name == "hiermem" and hasattr(system, 'constraint_store'):
        system_state = {
            "final_constraints": system.constraint_store.get_display_text(),
            "total_segments": len(system.archive.l0_directory),
            "l0_directory": system.archive.get_l0_directory_text(),
            "conversation_trace": system.get_conversation_trace(),
        }
    elif hasattr(system, 'turn_count'):
        system_state = {"turns_processed": system.turn_count}
    
    return {
        "conversation_id": conversation.get("conversation_id", "unknown"),
        "system": system_name,
        "turns_processed": len(turn_logs),
        "total_elapsed_seconds": total_elapsed,
        "avg_latency_per_turn": round(total_elapsed / max(len(turn_logs), 1), 2),
        "turn_logs": turn_logs,
        "checkpoints": checkpoints,
        "system_state": system_state,
        "config": {
            "provider": config.DEFAULT_PROVIDER,
            "main_model": config.MAIN_LLM_MODEL,
            "curator_model": config.CURATOR_MODEL,
            "summarizer_model": config.SUMMARIZER_MODEL,
            "context_budget": config.TOTAL_CONTEXT_BUDGET,
        },
    }


def run_benchmark(systems_to_run: List[str], benchmark_name: str,
                  output_dir: Path = None, max_convos: int = None):
    """Run specified systems on a benchmark."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_model = config.MAIN_LLM_MODEL.split("/")[-1].replace(":", "-")
    default_dir = config.RESULTS_PATH / "raw" / "benchmarks" / f"{benchmark_name}_{config.DEFAULT_PROVIDER}_{main_model}_{timestamp}"
    output_dir = output_dir or default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conversations = load_benchmark(benchmark_name)
    if max_convos:
        conversations = conversations[:max_convos]
    print(f"Loaded {len(conversations)} conversations from '{benchmark_name}'")
    
    all_results = {}
    
    for system_name in systems_to_run:
        if system_name not in SYSTEMS:
            print(f"Unknown system: {system_name}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running: {system_name} | Provider: {config.DEFAULT_PROVIDER}")
        print(f"Models: main={config.MAIN_LLM_MODEL}, curator={config.CURATOR_MODEL}")
        print(f"{'='*60}")
        
        system_results = []
        
        for conv in tqdm(conversations, desc=system_name):
            # Create fresh system instance for each conversation
            if system_name == "hiermem":
                system = HierMemPipeline.create()
            else:
                system = SYSTEMS[system_name]()
            
            result = run_system_on_conversation(system, system_name, conv)
            system_results.append(result)
            
            # Save after each conversation (in case of crash)
            partial_file = output_dir / f"{system_name}_partial.json"
            partial_file.write_text(json.dumps(system_results, indent=2))
        
        all_results[system_name] = system_results
        
        # Save per-system detailed results
        system_file = output_dir / f"{system_name}_detailed.json"
        system_file.write_text(json.dumps(system_results, indent=2))
        print(f"  Detailed results: {system_file}")
    
    # Save combined results
    results_file = output_dir / "results.json"
    results_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results saved to {results_file}")
    
    # Compute metrics (with LLM judge if available)
    print("\nInitializing LLM judge for evaluation...")
    init_llm_judge()
    metrics = compute_all_metrics(all_results)
    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_file}")
    
    # Save run config
    run_info = {
        "timestamp": timestamp,
        "benchmark": benchmark_name,
        "systems": systems_to_run,
        "num_conversations": len(conversations),
        "provider": config.DEFAULT_PROVIDER,
        "models": {
            "main": config.MAIN_LLM_MODEL,
            "curator": config.CURATOR_MODEL,
            "summarizer": config.SUMMARIZER_MODEL,
        },
        "config": {
            "context_budget": config.TOTAL_CONTEXT_BUDGET,
            "max_l0_entries": config.MAX_L0_ENTRIES,
            "segment_size": config.SEGMENT_SIZE,
            "max_constraints": config.MAX_CONSTRAINTS,
        }
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    
    # Clean up partial files
    for f in output_dir.glob("*_partial.json"):
        f.unlink()
    
    return all_results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HierMem benchmarks")
    parser.add_argument("--systems", nargs="+", default=["all"],
                        help="Systems to evaluate (or 'all')")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark name (directory in eval/datasets/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--max-convos", type=int, default=None,
                        help="Limit number of conversations (for quick testing)")
    parser.add_argument("--rescore", type=str, default=None,
                        help="Re-score existing results directory (skip running, just re-evaluate)")
    args = parser.parse_args()

    if args.rescore:
        # Re-score mode: load existing results and recompute metrics
        rescore_dir = Path(args.rescore)
        results_file = rescore_dir / "results.json"
        if not results_file.exists():
            print(f"Error: {results_file} not found")
            exit(1)
        all_results = json.loads(results_file.read_text(encoding="utf-8"))
        print(f"Re-scoring results from {rescore_dir}")
        print("Initializing LLM judge...")
        init_llm_judge()

        # Debug: show checkpoint evaluation details
        from eval.metrics import evaluate_checkpoint, _get_results, _find_response_at_turn, _deduplicate_checkpoints, _get_constraint_text
        for sys_name, convos in all_results.items():
            print(f"\n--- {sys_name} ---")
            for ci, conv in enumerate(convos):
                checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
                results = _get_results(conv)
                constraint_text = _get_constraint_text(conv)
                print(f"  Convo {ci}: {len(checkpoints)} checkpoints, {len(results)} turns, constraint_text={constraint_text[:80]}...")
                for cp in checkpoints:
                    response = _find_response_at_turn(results, cp.get("turn", 0))
                    if response and not response.startswith("ERROR:"):
                        passed = evaluate_checkpoint(
                            response, constraint_text,
                            cp.get("constraint_tested", ""),
                            cp.get("keywords", [])
                        )
                        print(f"    Turn {cp['turn']}: {'PASS' if passed else 'FAIL'} (keywords={cp.get('keywords',[])})")
                        print(f"      Response preview: {response[:120]}...")

        metrics = compute_all_metrics(all_results)
        metrics_file = rescore_dir / "metrics_rescored.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))
        print(f"\nRe-scored metrics saved to {metrics_file}")
        print(json.dumps(metrics, indent=2))
    else:
        systems = list(SYSTEMS.keys()) if "all" in args.systems else args.systems
        output_dir = Path(args.output) if args.output else None
        run_benchmark(systems, args.benchmark, output_dir, args.max_convos)

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


def run_system_on_conversation(system, system_name: str, conversation: Dict,
                               conv_index: int = 0, total_convos: int = 1) -> Dict:
    """Run a single system on a single conversation with detailed per-step logging."""
    turns = conversation.get("turns", [])
    checkpoints = conversation.get("checkpoints", [])
    user_turns = [t for t in turns if t.get("role") == "user"]
    total_user_turns = len(user_turns)
    turn_logs = []
    total_start = time.time()
    conv_id = conversation.get("conversation_id", "unknown")
    
    print(f"\n  [{conv_index+1}/{total_convos}] {conv_id} ({total_user_turns} turns)")
    
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
            
            turn_elapsed = round(time.time() - turn_start, 2)
            total_elapsed = round(time.time() - total_start, 2)
            
            # Live progress line
            mode = ""
            if system_name == "hiermem" and hasattr(result, 'warnings'):
                if any("passthrough" in w for w in result.warnings):
                    mode = " [pass]"
                elif hasattr(result, 'curator_decision') and result.curator_decision:
                    mode = f" [{result.curator_decision.retrieval_strategy}]"
            tokens = result.tokens_used if hasattr(result, 'tokens_used') else \
                     result.context_tokens if hasattr(result, 'context_tokens') else ""
            tokens_str = f" {tokens}tok" if tokens else ""
            
            # Estimate remaining time
            turns_done = len(turn_logs) + 1
            avg_per_turn = total_elapsed / turns_done
            turns_left = total_user_turns - turns_done
            eta_mins = round(avg_per_turn * turns_left / 60, 1)
            
            print(f"    Turn {turn_num:2d}/{total_user_turns} | {turn_elapsed:5.1f}s{mode}{tokens_str} | "
                  f"ETA: {eta_mins}min", flush=True)
            
            # Build detailed turn log
            turn_log = {
                "turn": turn_num,
                "type": turn_type,
                "user": user_msg,
                "response": response,
                "latency_seconds": turn_elapsed,
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
            turn_elapsed = round(time.time() - turn_start, 2)
            print(f"    Turn {turn_num:2d}/{total_user_turns} | ERROR: {e}", flush=True)
            turn_log = {
                "turn": turn_num,
                "type": turn_type,
                "user": user_msg,
                "response": response,
                "latency_seconds": turn_elapsed,
                "error": str(e),
            }
        
        turn_logs.append(turn_log)
    
    total_elapsed = round(time.time() - total_start, 2)
    print(f"  ✓ {conv_id} done in {total_elapsed/60:.1f}min (avg {total_elapsed/max(len(turn_logs),1):.1f}s/turn)")
    
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
                  output_dir: Path = None, max_convos: int = None,
                  convo_ids: List[str] = None, run_dir: Path = None,
                  judge_provider: str = None, judge_model: str = None):
    """Run specified systems on a benchmark.
    
    Args:
        run_dir: If set, accumulate results into this directory across multiple runs.
                 New results are merged with existing ones.
    """
    main_model = config.MAIN_LLM_MODEL.split("/")[-1].replace(":", "-")
    
    if run_dir:
        output_dir = run_dir
    elif output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.RESULTS_PATH / "raw" / "benchmarks" / f"{benchmark_name}_{config.DEFAULT_PROVIDER}_{main_model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conversations = load_benchmark(benchmark_name)
    
    # Filter by specific conversation IDs
    if convo_ids:
        conversations = [c for c in conversations 
                        if c.get("conversation_id", "") in convo_ids]
        if not conversations:
            print(f"No conversations found matching: {convo_ids}")
            print("Use --list to see available conversation IDs")
            return None, None
    elif max_convos:
        conversations = conversations[:max_convos]
    
    print(f"Loaded {len(conversations)} conversations from '{benchmark_name}'")
    for c in conversations:
        print(f"  - {c.get('conversation_id', '?')} ({len([t for t in c.get('turns',[]) if t.get('role')=='user'])} turns)")
    
    # Load existing results if accumulating
    all_results = {}
    if run_dir:
        results_file = output_dir / "results.json"
        if results_file.exists():
            all_results = json.loads(results_file.read_text(encoding="utf-8"))
            existing_count = sum(len(v) for v in all_results.values())
            print(f"Loaded {existing_count} existing results from {results_file}")
    
    for system_name in systems_to_run:
        if system_name not in SYSTEMS:
            print(f"Unknown system: {system_name}, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running: {system_name} | Provider: {config.DEFAULT_PROVIDER}")
        print(f"Models: main={config.MAIN_LLM_MODEL}, curator={config.CURATOR_MODEL}")
        print(f"{'='*60}")
        
        # Get existing results for this system
        system_results = list(all_results.get(system_name, []))
        existing_ids = {r["conversation_id"] for r in system_results}
        
        # Skip conversations already done
        convos_to_run = [c for c in conversations 
                        if c.get("conversation_id", "") not in existing_ids]
        
        if not convos_to_run:
            print(f"  All conversations already done for {system_name}, skipping")
            continue
        
        if len(convos_to_run) < len(conversations):
            print(f"  Skipping {len(conversations) - len(convos_to_run)} already-completed conversations")
        
        for i, conv in enumerate(convos_to_run):
            # Create fresh system instance for each conversation
            if system_name == "hiermem":
                system = HierMemPipeline.create()
            else:
                system = SYSTEMS[system_name]()
            
            result = run_system_on_conversation(system, system_name, conv,
                                                conv_index=i, total_convos=len(convos_to_run))
            system_results.append(result)
            
            # Save after each conversation (crash-safe)
            all_results[system_name] = system_results
            results_file = output_dir / "results.json"
            results_file.write_text(json.dumps(all_results, indent=2))
        
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
    init_llm_judge(provider=judge_provider, model=judge_model)
    metrics = compute_all_metrics(all_results)
    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_file}")
    
    # Quick metrics summary
    print("\n" + "="*50)
    for sys_name, m in metrics.items():
        if sys_name == "pairwise_comparison":
            continue
        js = m.get("judge_scores", {})
        print(f"  {sys_name}: judge={js.get('avg_score','?')}/10  "
              f"accuracy={m['task_accuracy']*100:.0f}%")
    if "pairwise_comparison" in metrics:
        pc = metrics["pairwise_comparison"]
        print(f"  Pairwise: {pc['system_a']}={pc['wins_a']}W "
              f"{pc['system_b']}={pc['wins_b']}W ties={pc['ties']}")
    print("="*50)
    
    # Save run config
    run_info = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
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
    parser.add_argument("--convo", nargs="+", default=None,
                        help="Run specific conversation IDs (e.g. --convo chatgpt_cooking_recipes_01)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Accumulate results into this directory across multiple runs")
    parser.add_argument("--list", action="store_true",
                        help="List available conversations and exit")
    parser.add_argument("--rescore", type=str, default=None,
                        help="Re-score existing results directory (skip running, just re-evaluate)")
    parser.add_argument("--judge-provider", type=str, default=None,
                        help="Provider for LLM judge (e.g. google, openai). Default: same as benchmark provider")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model for LLM judge (e.g. gemini-2.5-flash, gpt-4o-mini). Default: summarizer model")
    args = parser.parse_args()

    if args.list:
        conversations = load_benchmark(args.benchmark)
        print(f"Available conversations in '{args.benchmark}' ({len(conversations)} total):\n")
        for i, c in enumerate(conversations):
            cid = c.get("conversation_id", "unknown")
            user_turns = len([t for t in c.get("turns", []) if t.get("role") == "user"])
            checkpoints = len(c.get("checkpoints", []))
            print(f"  {i:2d}. {cid} ({user_turns} turns, {checkpoints} checkpoints)")
        exit(0)

    if args.rescore:
        # Re-score mode: load existing results and recompute metrics
        rescore_dir = Path(args.rescore)
        results_file = rescore_dir / "results.json"
        if not results_file.exists():
            print(f"Error: {results_file} not found")
            exit(1)
        all_results = json.loads(results_file.read_text(encoding="utf-8"))
        print(f"Re-scoring results from {rescore_dir}")
        
        judge_provider = args.judge_provider
        judge_model = args.judge_model
        if judge_provider:
            print(f"Using LLM judge: provider={judge_provider}, model={judge_model or 'default'}")
        print("Initializing LLM judge...")
        init_llm_judge(provider=judge_provider, model=judge_model)

        # Debug: show checkpoint evaluation details
        from eval.metrics import (evaluate_checkpoint, score_checkpoint,
            _get_results, _find_response_at_turn, _deduplicate_checkpoints,
            _get_constraint_text, _score_keyword_group, _absolute_judge)
        for sys_name, convos in all_results.items():
            print(f"\n--- {sys_name} ---")
            for ci, conv in enumerate(convos):
                cid = conv.get("conversation_id", f"conv_{ci}")
                checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
                results = _get_results(conv)
                constraint_text = _get_constraint_text(conv)
                print(f"  {cid}: {len(checkpoints)} checkpoints")
                for cp in checkpoints:
                    turn = cp.get("turn", 0)
                    keywords = cp.get("keywords", [])
                    kw_groups = cp.get("keyword_groups")
                    response = _find_response_at_turn(results, turn)
                    if response and not response.startswith("ERROR:"):
                        # Gradient score
                        sc = score_checkpoint(response, keywords, kw_groups)
                        group_strs = []
                        if kw_groups:
                            for g in kw_groups:
                                gs = _score_keyword_group(response, g)
                                group_strs.append(f"{gs:.0f}:{g[:3]}")
                        
                        passed = evaluate_checkpoint(
                            response, constraint_text,
                            cp.get("constraint_tested", ""),
                            keywords,
                            keyword_groups=kw_groups
                        )
                        print(f"    Turn {turn}: score={sc:.0f} ({'PASS' if passed else 'FAIL'}) "
                              f"groups=[{', '.join(group_strs)}]")

        metrics = compute_all_metrics(all_results)
        metrics_file = rescore_dir / "metrics_rescored.json"
        metrics_file.write_text(json.dumps(metrics, indent=2))
        print(f"\nRe-scored metrics saved to {metrics_file}")
        
        # Print summary — judge score (1-10) is the headline metric
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for sys_name, m in metrics.items():
            if sys_name == "pairwise_comparison":
                continue
            js = m.get("judge_scores", {})
            pts = m.get("per_turn_scores", {})
            print(f"\n  {sys_name}:")
            if js.get("avg_score") is not None:
                print(f"    Judge score (1-10): {js['avg_score']:.2f}  ← PRIMARY METRIC")
            print(f"    Keyword score:      {pts.get('avg_score', '?')}/100")
            print(f"    Accuracy (binary):  {m['task_accuracy']*100:.0f}%")
            print(f"    CVR:                {m['constraint_violation_rate']*100:.0f}%")
            for d in js.get("detail", []):
                # Match kw score by both turn AND convo to avoid cross-convo collision
                kw_score = next(
                    (dd["score"] for dd in pts.get("detail", [])
                     if dd["turn"] == d["turn"] and dd["convo"] == d["convo"]),
                    "?"
                )
                print(f"      [{d['convo']}] Turn {d['turn']:2d}: "
                      f"judge={d['judge_score']:.1f}/10  kw={kw_score}/100  ({d['mode']})")
        
        if "pairwise_comparison" in metrics:
            pc = metrics["pairwise_comparison"]
            print(f"\n  Pairwise ({pc['system_a']} vs {pc['system_b']}):")
            print(f"    {pc['system_a']} wins: {pc['wins_a']}/{pc['total']} "
                  f"({pc['win_rate_a']:.0f}%)")
            print(f"    {pc['system_b']} wins: {pc['wins_b']}/{pc['total']} "
                  f"({pc['win_rate_b']:.0f}%)")
            print(f"    ties: {pc['ties']}")
            for c in pc.get("comparisons", []):
                print(f"      Turn {c['turn']:2d}: A={c['score_a']} B={c['score_b']} "
                      f"→ {c['winner']} | {c.get('reasoning', '')}")
        print("="*60)
    else:
        systems = list(SYSTEMS.keys()) if "all" in args.systems else args.systems
        output_dir = Path(args.output) if args.output else None
        run_dir = Path(args.run_dir) if args.run_dir else None
        run_benchmark(systems, args.benchmark, output_dir, args.max_convos,
                      convo_ids=args.convo, run_dir=run_dir,
                      judge_provider=args.judge_provider, judge_model=args.judge_model)

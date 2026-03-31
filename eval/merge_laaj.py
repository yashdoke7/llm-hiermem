"""
Merge per-system LAAJ judge outputs into a single combined laaj.json.

Usage:
    python eval/merge_laaj.py results/raw/benchmarks/qwen14b_arch_c/dataset_se_03

Expects: laaj_hiermem.json, laaj_raw_llm.json, laaj_rag.json, laaj_rag_summary.json
in the dataset directory. These are individual judge outputs from per-system evaluation.

Produces: laaj.json (merged, with cross-system analysis computed).
"""

import json
import sys
from pathlib import Path


def extract_system_eval(laaj_data: dict) -> tuple:
    """Extract the system evaluation from a per-system LAAJ output."""
    se = laaj_data.get('system_evaluations', {})
    
    # Get the first (and only) system key
    if not se:
        return None, None
    
    sys_key = list(se.keys())[0]
    sys_eval = se[sys_key]
    
    # Resolve identity
    reveal = laaj_data.get('system_identity_reveal', {})
    real_name = reveal.get(sys_key, sys_key)
    
    return real_name, sys_eval


def compute_cross_system(system_evals: dict) -> dict:
    """Compute cross-system comparison from individual evaluations."""
    # Get all checkpoint turns from the first system
    first_sys = list(system_evals.values())[0]
    checkpoint_turns = [cp['checkpoint_turn'] for cp in first_sys.get('checkpoints', [])]
    
    # Checkpoint comparison
    checkpoint_comparison = []
    for ct in checkpoint_turns:
        scores = {}
        for sys_name, sys_eval in system_evals.items():
            for cp in sys_eval.get('checkpoints', []):
                if cp.get('checkpoint_turn') == ct:
                    scores[sys_name] = cp.get('weighted_score', 0)
                    break
        
        top_sys = max(scores, key=scores.get) if scores else "unknown"
        checkpoint_comparison.append({
            "checkpoint_turn": ct,
            "scores": scores,
            "top_system": top_sys,
        })
    
    # Final rankings
    avg_scores = {
        sys_name: sys_eval.get('average_score', 0)
        for sys_name, sys_eval in system_evals.items()
    }
    sorted_systems = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    rankings = {}
    for rank, (sys_name, score) in enumerate(sorted_systems, 1):
        rankings[str(rank)] = {
            "system": sys_name,
            "average_score": round(score, 2),
            "degradation_delta": system_evals[sys_name].get('degradation_delta', 0),
            "degradation_pattern": system_evals[sys_name].get('degradation_pattern', 'Unknown'),
        }
    
    # Degradation comparison
    deg_deltas = {
        sys_name: abs(sys_eval.get('degradation_delta', 0))
        for sys_name, sys_eval in system_evals.items()
    }
    
    return {
        "checkpoint_comparison": checkpoint_comparison,
        "final_rankings": rankings,
        "degradation_comparison": {
            "most_stable": min(deg_deltas, key=deg_deltas.get),
            "most_degraded": max(deg_deltas, key=deg_deltas.get),
        },
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval/merge_laaj.py <dataset_results_dir>")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    
    systems = ['hiermem', 'raw_llm', 'rag', 'rag_summary']
    system_evals = {}
    metadata = None
    constraint_registry = None
    validation = None
    
    for sys_name in systems:
        laaj_file = dataset_dir / f"laaj_{sys_name}.json"
        if not laaj_file.exists():
            print(f"  Skipping {sys_name}: {laaj_file.name} not found")
            continue
        
        try:
            data = json.load(open(laaj_file, encoding='utf-8'))
        except json.JSONDecodeError as e:
            print(f"  ERROR parsing {laaj_file.name}: {e}")
            continue
        
        real_name, sys_eval = extract_system_eval(data)
        if sys_eval is None:
            print(f"  WARNING: No system_evaluations in {laaj_file.name}")
            continue
        
        system_evals[real_name or sys_name] = sys_eval
        
        # Capture shared metadata from first valid file
        if metadata is None:
            metadata = data.get('evaluation_metadata', {})
            constraint_registry = data.get('constraint_registry', [])
            validation = data.get('validation_checks', {})
        
        print(f"  Loaded {sys_name}: avg={sys_eval.get('average_score', 0):.2f}")
    
    if len(system_evals) < 2:
        print("ERROR: Need at least 2 systems to merge. Check your laaj_*.json files.")
        sys.exit(1)
    
    # Build merged output
    merged = {
        "evaluation_metadata": metadata or {},
        "constraint_registry": constraint_registry or [],
        "system_evaluations": system_evals,
        "cross_system_analysis": compute_cross_system(system_evals),
        "validation_checks": validation or {},
        "system_identity_reveal": {
            sys_name: sys_name for sys_name in system_evals
        },
    }
    
    # Update metadata
    merged['evaluation_metadata']['systems_evaluated'] = list(system_evals.keys())
    merged['evaluation_metadata']['merge_note'] = "Merged from per-system evaluations"
    
    output_file = dataset_dir / "laaj.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    size_kb = output_file.stat().st_size / 1024
    print(f"\nMerged {len(system_evals)} systems -> {output_file} ({size_kb:.0f} KB)")
    
    # Print final rankings
    print("\nFinal Rankings:")
    for rank, info in sorted(merged['cross_system_analysis']['final_rankings'].items()):
        print(f"  #{rank}: {info['system']:15s} avg={info['average_score']:.2f}")


if __name__ == "__main__":
    main()

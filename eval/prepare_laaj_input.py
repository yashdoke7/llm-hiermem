"""
Prepare per-system LAAJ judge input — feeds ONE system at a time to the judge.

Usage:
    python eval/prepare_laaj_input.py results/raw/benchmarks/qwen14b_arch_c/dataset_se_03

This produces 4 files in the dataset directory:
    laaj_input_hiermem.json
    laaj_input_raw_llm.json
    laaj_input_rag.json
    laaj_input_rag_summary.json

Each file contains ONLY the dataset ground truth + that system's responses,
trimmed to reduce token count. Feed each file to the judge separately,
then use merge_laaj.py to combine the 4 outputs.
"""

import json
import sys
from pathlib import Path


def load_dataset_ground_truth(dataset_dir: Path) -> dict:
    """Load the matching dataset file for ground truth."""
    # Get conversation ID from any detailed file
    for f in dataset_dir.glob("*_detailed.json"):
        data = json.load(open(f, encoding='utf-8'))
        if isinstance(data, list):
            data = data[0]
        convo_id = data.get('conversation_id', data.get('source_file', ''))
        break
    
    # Search datasets
    ds_dir = Path(__file__).parent / "datasets" / "constraint_tracking" / "llm_generated"
    ds_file = dataset_dir.name + ".json"
    ds_path = ds_dir / ds_file
    
    if ds_path.exists():
        ds = json.load(open(ds_path, encoding='utf-8'))
        return ds[0] if isinstance(ds, list) else ds
    
    # Fallback: search all dataset files
    for f in ds_dir.glob("*.json"):
        ds = json.load(open(f, encoding='utf-8'))
        if isinstance(ds, list):
            ds = ds[0]
        if ds.get('conversation_id') == convo_id:
            return ds
    
    print(f"WARNING: No dataset ground truth found for {dataset_dir.name}")
    return {}


def get_checkpoint_turns(dataset: dict) -> set:
    """Get the set of checkpoint turn numbers from the dataset."""
    cps = dataset.get('checkpoints', [])
    return {cp.get('checkpoint_turn') for cp in cps}


def trim_turn_logs(turn_logs: list, checkpoint_turns: set) -> list:
    """Trim turn logs: full text at checkpoints, abbreviated elsewhere."""
    trimmed = []
    for t in turn_logs:
        turn_num = t.get('turn', t.get('turn_number', 0))
        
        entry = {
            "turn": turn_num,
            "role": t.get('role', 'unknown'),
        }
        
        # Always include user message (usually short)
        if t.get('user'):
            entry['user'] = t['user']
        if t.get('user_message'):
            entry['user'] = t['user_message']
        
        # For assistant responses
        response = t.get('assistant', t.get('response', ''))
        
        # Determine if this is a checkpoint turn (check both global and raw numbering)
        is_checkpoint = (turn_num in checkpoint_turns or 
                         turn_num + 1 in checkpoint_turns or
                         turn_num - 1 in checkpoint_turns)
        
        if response:
            if is_checkpoint:
                # Full response at checkpoints — judge needs the complete text
                entry['assistant'] = response
            else:
                # Abbreviated for non-checkpoints (context only)
                entry['assistant'] = response[:300] + "..." if len(response) > 300 else response
        
        # Strip pipeline internals that the judge doesn't need
        # Keep only: turn, role, user, assistant
        # Remove: curator_strategy, context_tokens, sources, warnings,
        #         pipeline_details, latency, etc.
        
        trimmed.append(entry)
    
    return trimmed


def prepare_single_system(dataset_dir: Path, system_name: str, 
                          dataset: dict, checkpoint_turns: set) -> dict:
    """Prepare judge input for a single system."""
    detailed_file = dataset_dir / f"{system_name}_detailed.json"
    
    if not detailed_file.exists():
        print(f"  Skipping {system_name}: no detailed file found")
        return None
    
    data = json.load(open(detailed_file, encoding='utf-8'))
    if isinstance(data, list):
        data = data[0]
    
    # Trim turn logs
    turn_logs = data.get('turn_logs', [])
    trimmed_logs = trim_turn_logs(turn_logs, checkpoint_turns)
    
    # Build the judge input
    system_input = {
        "conversation_id": data.get('conversation_id', data.get('source_file', '')),
        "system": system_name,
        "turn_logs": trimmed_logs,
    }
    
    # Build complete judge package
    judge_package = {
        "_instructions": (
            "Evaluate this SINGLE system against the constraint ground truth below. "
            "Follow the evaluation rubric from the system prompt exactly. "
            "Output a complete JSON evaluation for this one system only."
        ),
        "dataset_ground_truth": {
            "conversation_id": dataset.get('conversation_id', ''),
            "domain": dataset.get('domain', ''),
            "topic": dataset.get('topic', dataset.get('title', '')),
            "constraint_log": dataset.get('constraint_log', dataset.get('constraints', [])),
            "checkpoints": dataset.get('checkpoints', []),
            "total_turns": dataset.get('total_turns', len(turn_logs)),
        },
        "system_data": system_input,
        "system_key_map": {f"system_a": system_name},
    }
    
    return judge_package


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval/prepare_laaj_input.py <dataset_results_dir>")
        print("Example: python eval/prepare_laaj_input.py results/raw/benchmarks/qwen14b_arch_c/dataset_se_03")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} is not a directory")
        sys.exit(1)
    
    # Load ground truth
    dataset = load_dataset_ground_truth(dataset_dir)
    checkpoint_turns = get_checkpoint_turns(dataset)
    
    print(f"Dataset: {dataset_dir.name}")
    print(f"Checkpoint turns: {sorted(checkpoint_turns)}")
    print(f"Constraints: {len(dataset.get('constraint_log', dataset.get('constraints', [])))}")
    
    systems = ['hiermem', 'raw_llm', 'rag', 'rag_summary']
    
    for sys_name in systems:
        package = prepare_single_system(dataset_dir, sys_name, dataset, checkpoint_turns)
        if package is None:
            continue
        
        output_file = dataset_dir / f"laaj_input_{sys_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(package, f, indent=2, ensure_ascii=False)
        
        # Report size
        size_kb = output_file.stat().st_size / 1024
        n_turns = len(package['system_data']['turn_logs'])
        print(f"  {sys_name}: {output_file.name} ({size_kb:.0f} KB, {n_turns} turns)")
    
    print(f"\nDone! Feed each laaj_input_*.json to the judge separately.")
    print(f"Then run: python eval/merge_laaj.py {dataset_dir}")


if __name__ == "__main__":
    main()

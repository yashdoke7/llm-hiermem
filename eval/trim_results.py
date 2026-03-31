"""
Trim results JSON files — remove pipeline internals not needed for LAAJ judging.

Usage:
    python eval/trim_results.py results/raw/benchmarks/qwen14b_arch_c/dataset_se_03

This modifies the *_detailed.json files IN PLACE (backs up originals as *_detailed.full.json).
Removes: curator_strategy, context_tokens, sources, warnings, pipeline_details,
         assembled_context, latency, curator_decision fields from turn_logs.

Keeps: turn, role, user/user_message, assistant/response, type, checkpoint data.
"""

import json
import sys
import shutil
from pathlib import Path


# Fields to REMOVE from turn_logs entries (pipeline internals)
STRIP_FIELDS = {
    'curator_strategy', 'context_tokens', 'sources', 'warnings',
    'pipeline_details', 'assembled_context', 'latency', 'latency_seconds',
    'curator_decision', 'retrieval_strategy', 'tokens_used',
    'context_tokens_used', 'passthrough_mode', 'system_state',
    'memory_state', 'constraint_store_state', 'archive_state',
}

# Fields to KEEP (whitelist approach for safety)
KEEP_FIELDS = {
    'turn', 'turn_number', 'role', 'type',
    'user', 'user_message', 'assistant', 'response',
    'checkpoint', 'checkpoint_turn', 'active_constraints',
}


def trim_turn(turn: dict, is_checkpoint: bool = False) -> dict:
    """Trim a single turn entry."""
    trimmed = {}
    
    for key, value in turn.items():
        if key in STRIP_FIELDS:
            continue  # Remove pipeline internals
        trimmed[key] = value
    
    return trimmed


def trim_detailed_file(filepath: Path, checkpoint_turns: set = None):
    """Trim a single detailed results file."""
    data = json.load(open(filepath, encoding='utf-8'))
    is_list = isinstance(data, list)
    items = data if is_list else [data]
    
    original_size = filepath.stat().st_size
    
    for item in items:
        turn_logs = item.get('turn_logs', [])
        trimmed_logs = []
        
        for t in turn_logs:
            turn_num = t.get('turn', t.get('turn_number', 0))
            is_cp = checkpoint_turns and turn_num in checkpoint_turns
            trimmed = trim_turn(t, is_checkpoint=is_cp)
            trimmed_logs.append(trimmed)
        
        item['turn_logs'] = trimmed_logs
        
        # Also strip top-level system_state and config if present
        for strip_key in ['system_state', 'config']:
            item.pop(strip_key, None)
    
    output = data if is_list else items[0]
    
    # Save
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    new_size = filepath.stat().st_size
    reduction = (1 - new_size / original_size) * 100 if original_size > 0 else 0
    
    return original_size, new_size, reduction


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval/trim_results.py <dataset_results_dir> [--no-backup]")
        print("Example: python eval/trim_results.py results/raw/benchmarks/qwen14b_arch_c/dataset_se_03")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    no_backup = '--no-backup' in sys.argv
    
    if not dataset_dir.is_dir():
        print(f"Error: {dataset_dir} is not a directory")
        sys.exit(1)
    
    # Find all detailed files
    detailed_files = list(dataset_dir.glob("*_detailed.json"))
    if not detailed_files:
        print(f"No *_detailed.json files found in {dataset_dir}")
        sys.exit(1)
    
    # Load checkpoint turns from dataset ground truth (optional)
    checkpoint_turns = set()
    ds_dir = Path(__file__).parent / "datasets" / "constraint_tracking" / "llm_generated"
    ds_file = ds_dir / (dataset_dir.name + ".json")
    if ds_file.exists():
        ds = json.load(open(ds_file, encoding='utf-8'))
        if isinstance(ds, list):
            ds = ds[0]
        for cp in ds.get('checkpoints', []):
            checkpoint_turns.add(cp.get('checkpoint_turn'))
    
    print(f"Trimming {len(detailed_files)} files in {dataset_dir.name}")
    if checkpoint_turns:
        print(f"Checkpoint turns: {sorted(checkpoint_turns)}")
    
    total_saved = 0
    
    for filepath in detailed_files:
        # Backup original
        if not no_backup:
            backup_path = filepath.with_suffix('.full.json')
            if not backup_path.exists():
                shutil.copy2(filepath, backup_path)
        
        orig_size, new_size, reduction = trim_detailed_file(filepath, checkpoint_turns)
        total_saved += (orig_size - new_size)
        
        print(f"  {filepath.name}: {orig_size/1024:.0f}KB -> {new_size/1024:.0f}KB ({reduction:.0f}% reduction)")
    
    print(f"\nTotal saved: {total_saved/1024:.0f} KB ({total_saved/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()

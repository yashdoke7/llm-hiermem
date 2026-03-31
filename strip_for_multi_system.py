#!/usr/bin/env python3
"""
strip_for_multi_system.py
=========================
Parses the Raw LLM Output logs and generates a single compact
Multi-System Payload per dataset.

This takes all 4 system runs for a dataset, extracts ONLY 
the checkpoints and constraints, and produces one highly optimised file:
`multi_judge_input_<dataset>.json`

Usage:
  # All datasets
  python strip_for_multi_system.py

  # Single dataset
  python strip_for_multi_system.py --dataset_id dataset_se_01
"""

import json
import argparse
import random
import os
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DATASETS_DIR  = Path("eval/datasets")
DEFAULT_RESULTS_ROOT  = Path("results/raw/benchmarks/qwen14b_arch_c")
SYSTEMS = ["hiermem", "raw_llm", "rag", "rag_summary"]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ─── LOGIC (From pairwise stripper) ──────────────────────────────────────────

def normalize_constraint_log(raw_log: list) -> list:
    unified_log = []
    if raw_log and "introduced_turn" in raw_log[0] and "id" in raw_log[0]:
        for c in raw_log:
            cid = c["id"]
            unified_log.append({"turn": c["introduced_turn"], "constraint_id": cid,
                                 "event": "add",
                                 "constraint_text": c.get("original_text", c.get("final_text", ""))})
            for lc in c.get("lifecycle", []):
                if lc.startswith("modified@"):
                    try:
                        m_turn = int(lc.split("@")[1])
                        unified_log.append({"turn": m_turn, "constraint_id": cid,
                                            "event": "modify",
                                            "constraint_text": c.get("final_text", "")})
                    except Exception:
                        pass
            if c.get("removed_turn"):
                unified_log.append({"turn": c["removed_turn"], "constraint_id": cid,
                                    "event": "remove", "constraint_text": ""})
        return unified_log

    for e in raw_log:
        turn = e.get("turn", e.get("event_turn"))
        cid  = e.get("constraint_id", e.get("id", "unknown"))
        etype = e.get("event", e.get("action", e.get("event_type", "add")))
        if etype == "update": etype = "modify"
        text = e.get("constraint_text", e.get("new_constraint_text", e.get("text", "")))
        if turn is not None and cid:
            unified_log.append({"turn": turn, "constraint_id": cid,
                                 "event": etype, "constraint_text": text})
    return unified_log

def resolve_constraint_at_turn(constraint_log: list, cid: str, cp_turn: int):
    current_text = None
    removed = False
    for event in sorted([e for e in constraint_log if e["constraint_id"] == cid],
                        key=lambda e: e["turn"]):
        if event["turn"] > cp_turn:
            break
        t = event.get("constraint_text", "")
        if event["event"] == "add":
            current_text = t; removed = False
        elif event["event"] == "modify":
            current_text = t
        elif event["event"] == "remove":
            removed = True; current_text = None
    return None if removed else current_text

def get_active_constraints_at_turn(constraint_log: list, cp_turn: int) -> dict:
    all_ids = list(dict.fromkeys(e["constraint_id"] for e in constraint_log))
    return {cid: text for cid in all_ids
            if (text := resolve_constraint_at_turn(constraint_log, cid, cp_turn)) is not None}

def build_constraint_registry(constraint_log: list) -> list:
    all_ids = list(dict.fromkeys(e["constraint_id"] for e in constraint_log))
    registry = []
    for cid in all_ids:
        events = sorted([e for e in constraint_log if e["constraint_id"] == cid],
                        key=lambda e: e["turn"])
        introduced_turn = removed_turn = original_text = final_text = None
        modification_turns = []
        for event in events:
            t = event.get("constraint_text", "")
            if event["event"] == "add":
                introduced_turn = event["turn"]; original_text = final_text = t
            elif event["event"] == "modify":
                modification_turns.append(event["turn"]); final_text = t
            elif event["event"] == "remove":
                removed_turn = event["turn"]
        status = "removed" if removed_turn else (
                 "active_modified" if modification_turns else "active")
        registry.append({"id": cid, "introduced_turn": introduced_turn,
                          "removed_turn": removed_turn,
                          "modification_turns": modification_turns,
                          "original_text": original_text, "final_text": final_text,
                          "status_at_end": status})
    return registry

def normalize_transcript(transcript_raw, dataset_turns: list) -> dict:
    dataset_asst_turns = sorted(t["turn"] for t in dataset_turns if t["role"] == "assistant")
    assistant_map: dict[int, str] = {}
    if isinstance(transcript_raw, list) and len(transcript_raw) > 0 and isinstance(transcript_raw[0], dict) and "turn_logs" in transcript_raw[0]:
        transcript_raw = transcript_raw[0]
    if isinstance(transcript_raw, dict) and "turn_logs" in transcript_raw:
        for entry in transcript_raw["turn_logs"]:
            odd_turn = entry.get("turn"); response = entry.get("response")
            if odd_turn is None or response is None: continue
            even_turn = odd_turn + 1
            if even_turn in dataset_asst_turns:
                assistant_map[even_turn] = response
            else:
                idx = (odd_turn - 1) // 2
                if idx < len(dataset_asst_turns):
                    assistant_map[dataset_asst_turns[idx]] = response
        return assistant_map
    if isinstance(transcript_raw, dict) and "turns" in transcript_raw:
        for entry in transcript_raw["turns"]:
            if entry.get("role") == "assistant":
                turn = entry.get("turn"); text = entry.get("text") or entry.get("response")
                if turn is not None and text is not None:
                    assistant_map[turn] = text
        return assistant_map
    if isinstance(transcript_raw, list):
        entries = transcript_raw
        if entries and "turn" in entries[0]:
            entries = sorted(entries, key=lambda e: e["turn"])
        for idx, entry in enumerate(entries):
            text = entry.get("text") or entry.get("response")
            if text is not None and idx < len(dataset_asst_turns):
                assistant_map[dataset_asst_turns[idx]] = text
        return assistant_map
    raise ValueError("Unrecognised transcript format.")

def get_user_request(dataset_def: dict, cp_turn: int) -> str:
    for t in dataset_def.get("turns", []):
        if t["turn"] == cp_turn - 1 and t["role"] == "user":
            return t["text"]
    return "[User prompt not found]"

# ─── MAIN BUILDER ────────────────────────────────────────────────────────────

def process_dataset(dataset_id: str, datasets_dir: Path, results_root: Path):
    ddef_path = datasets_dir / f"{dataset_id}.json"
    if not ddef_path.exists():
        ddef_path = list(datasets_dir.rglob(f"{dataset_id}.json"))[0]
    dataset_def = load_json(ddef_path)
    
    # Setup Random System Identity Reveal for THIS dataset
    shuffled_systems = SYSTEMS[:]
    random.shuffle(shuffled_systems)
    identity_reveal = {
        "system_a": shuffled_systems[0],
        "system_b": shuffled_systems[1],
        "system_c": shuffled_systems[2],
        "system_d": shuffled_systems[3]
    }
    
    constraint_log = normalize_constraint_log(dataset_def.get("constraint_log", []))
    checkpoints = [t["turn"] + 1 for t in dataset_def.get("turns", []) if t.get("is_checkpoint")]
    if not checkpoints:
        checkpoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    sys_transcripts = {}
    for sys_name in SYSTEMS:
        log_path = results_root / dataset_id / f"{sys_name}_detailed.json"
        if log_path.exists():
            raw_data = load_json(log_path)
            if isinstance(raw_data, list) and len(raw_data) > 0:
                raw_data = raw_data[0]
            # normalize_transcript checks for "turn_logs" or "turns" directly
            sys_transcripts[sys_name] = normalize_transcript(raw_data, dataset_def.get("turns", []))
        else:
            sys_transcripts[sys_name] = {}
            print(f"Warning: {log_path} not found.")

    multi_checkpoints = []
    
    last_cp = 0
    for cp_turn in checkpoints:
        active_constraints = get_active_constraints_at_turn(constraint_log, cp_turn)
        
        intermediate_history = []
        for t_info in dataset_def.get("turns", []):
            turn_num = t_info["turn"]
            # Capture ONLY the user's intermediate dialogue between checkpoints. 
            # Injecting 4 massive system responses per turn causes severe context overlap
            # and hallucination in the AI Studio judge.
            if last_cp < turn_num < cp_turn - 1:
                if t_info.get("role") == "user":
                    intermediate_history.append({
                        "turn": turn_num, 
                        "role": "user", 
                        "text": t_info.get("text", "")
                    })
        
        last_cp = cp_turn

        responses = {}
        for anon_key, actual_sys in identity_reveal.items():
            responses[anon_key] = sys_transcripts[actual_sys].get(cp_turn, "[MISSING]")
            
        multi_checkpoints.append({
            "checkpoint_turn": cp_turn,
            "intermediate_history_since_last_cp": intermediate_history,
            "user_request_prior": get_user_request(dataset_def, cp_turn),
            "active_constraints": active_constraints,
            "system_responses": responses
        })

    payload = {
        "evaluation_metadata": {
            "dataset_id": dataset_id,
        },
        "constraint_registry": build_constraint_registry(constraint_log),
        "checkpoints": multi_checkpoints,
        "system_identity_reveal": identity_reveal
    }

    out_path = results_root / dataset_id / f"multi_judge_input_{dataset_id}.json"
    save_json(out_path, payload)
    print(f"✓ Created: {out_path.relative_to(results_root.parent)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", help="Process only this dataset")
    parser.add_argument("--datasets_dir", default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    results_root = Path(args.results_root)

    print(f"Building Multi-System Inputs...")
    print(f"Results Root: {results_root}")
    print(f"Datasets Dir: {datasets_dir}\n")

    if args.dataset_id:
        process_dataset(args.dataset_id, datasets_dir, results_root)
    else:
        for dataset_dir in sorted(results_root.glob("dataset_*")):
            if not dataset_dir.is_dir(): continue
            process_dataset(dataset_dir.name, datasets_dir, results_root)

if __name__ == "__main__":
    main()

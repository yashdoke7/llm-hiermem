"""
Add more checkpoint turns to conversations that have too few.

Target: checkpoints at turns 5, 10, 15, 20, 25, 30 (6 per conversation).
Existing checkpoints are preserved; new ones are added at missing turns.

Usage:
    python -m eval.generators.add_checkpoints
"""

import json
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "constraint_tracking" / "llm_generated"
TARGET_TURNS = [5, 10, 15, 20, 25, 30]


def add_checkpoints_to_file(filepath: Path):
    data = json.loads(filepath.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    modified = False
    for convo in data:
        checkpoints = convo.get("checkpoints", [])
        existing_turns = {cp["turn"] for cp in checkpoints}
        constraints_text = convo.get("constraints", [""])[0]

        if not constraints_text:
            continue

        # Get keyword groups from existing checkpoints
        keyword_groups = []
        for cp in checkpoints:
            kws = cp.get("keywords", [])
            if kws and kws not in keyword_groups:
                keyword_groups.append(kws)

        if not keyword_groups:
            continue

        # Add missing checkpoint turns
        for target_turn in TARGET_TURNS:
            if target_turn not in existing_turns and target_turn <= convo.get("num_turns", 30):
                # Cycle through keyword groups
                for i, kw_group in enumerate(keyword_groups):
                    checkpoints.append({
                        "turn": target_turn,
                        "constraint_tested": constraints_text,
                        "test": f"Does response follow: '{constraints_text}'?",
                        "answer": True,
                        "keywords": kw_group,
                    })
                modified = True

        # Sort checkpoints by turn
        checkpoints.sort(key=lambda c: (c["turn"], str(c.get("keywords", []))))
        convo["checkpoints"] = checkpoints
        convo["num_checkpoints"] = len(checkpoints)

    if modified:
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        cp_count = sum(len(c.get("checkpoints", [])) for c in data)
        turns = sorted(set(cp["turn"] for c in data for cp in c.get("checkpoints", [])))
        print(f"  Updated: {filepath.name} -> {cp_count} checkpoints at turns {turns}")
    else:
        cp_count = sum(len(c.get("checkpoints", [])) for c in data)
        turns = sorted(set(cp["turn"] for c in data for cp in c.get("checkpoints", [])))
        print(f"  OK:      {filepath.name} -> {cp_count} checkpoints at turns {turns}")


def main():
    print(f"Adding checkpoints to conversations in {DATASET_DIR}")
    print(f"Target checkpoint turns: {TARGET_TURNS}\n")

    for filepath in sorted(DATASET_DIR.glob("*.json")):
        add_checkpoints_to_file(filepath)

    print("\nDone!")


if __name__ == "__main__":
    main()

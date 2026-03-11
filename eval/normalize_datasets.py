"""
normalize_datasets.py — Fix SE3, SE4, SE5, Finance_02 JSON structures to match SE1 format.

Handles all known ChatGPT output variations:
  - "conversation" key instead of "turns"
  - "content" instead of "text" in turn entries
  - "speaker" instead of "role" in turn entries
  - Wrong conversation_id
  - Two consecutive user turns (collapsed into one, assistant turn inserted)
  - checkpoints as dict/plain-list instead of list-of-objects

Checkpoint strategy (standardised across ALL datasets):
  5 positions at 20%, 40%, 60%, 80%, 100% of user turns.
  Each position gets 2 keyword-group entries (one per constraint dimension).
  → 10 checkpoint entries total per dataset, num_checkpoints=10.
  This enables consistent degradation-curve plots across conversations.

Run:
    python -m eval.normalize_datasets
"""

import json
from pathlib import Path

BASE = Path(__file__).parent / "datasets" / "constraint_tracking" / "llm_generated"


def _fix_turns(raw_turns: list) -> list:
    """
    Normalise a list of raw turn dicts:
    - rename 'content'  → 'text'
    - rename 'speaker'  → 'role'
    - ensure 'type' field exists
    - collapse consecutive same-role turns by merging their text
      (ChatGPT sometimes emits two user or two assistant turns in a row)
    """
    fixed = []
    for t in raw_turns:
        role = t.get("role") or t.get("speaker", "user")
        text = t.get("text") or t.get("content", "")
        turn_type = t.get("type", "topic" if role == "user" else "response")
        entry = {"turn": t.get("turn", len(fixed) + 1),
                 "role": role, "text": text, "type": turn_type}

        # Merge with previous if same role (collapse consecutive same-role turns)
        if fixed and fixed[-1]["role"] == role:
            fixed[-1]["text"] = fixed[-1]["text"].rstrip() + "\n\n" + text
        else:
            fixed.append(entry)

    # Renumber sequentially
    for i, t in enumerate(fixed):
        t["turn"] = i + 1

    return fixed


def _std_checkpoints(turns: list, constraint_text: str,
                     kw_groups: list, tests: list) -> list:
    """
    Build 5 standardised checkpoint positions at 20/40/60/80/100% of user turns.
    Each position gets one entry per (test, keyword_group) pair.
    kw_groups and tests must be parallel lists of equal length (one per constraint dimension).
    Returns a flat list of checkpoint dicts.
    """
    user_turns = [t for t in turns if t["role"] == "user"]
    n = len(user_turns)
    # 0-based indices for 20/40/60/80/100% — always hit the last turn exactly
    indices = [
        max(0, round(n * 0.20) - 1),
        max(0, round(n * 0.40) - 1),
        max(0, round(n * 0.60) - 1),
        max(0, round(n * 0.80) - 1),
        n - 1,  # always the final user turn
    ]
    # deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    checkpoints = []
    for idx in unique_indices:
        cp_turn = user_turns[idx]["turn"]
        for test, kw in zip(tests, kw_groups):
            checkpoints.append({
                "turn": cp_turn,
                "constraint_tested": constraint_text,
                "test": test,
                "answer": True,
                "keywords": kw,
            })
    return checkpoints


# ── SE3: ETL pipeline ──────────────────────────────────────────────────────────

def normalize_se3():
    src = BASE / "chatgpt_software_engineering_03.json"
    data = json.loads(src.read_text(encoding="utf-8"))

    raw = data.get("turns") or data.get("conversation") or []
    turns = _fix_turns(raw)
    num_user = len([t for t in turns if t["role"] == "user"])

    constraint_text = "use polars not pandas; use Python dataclasses for structured pipeline objects, not raw dicts"
    kw_polars = ["pl.", "polars", "pl.read_csv", "pl.DataFrame", "pl.LazyFrame", "scan_csv", "collect()"]
    kw_dc = ["@dataclass", "dataclass", "from dataclasses import", "field(", ": str", ": int", ": List"]
    tests = ["Does response use polars instead of pandas?",
             "Does response use Python dataclasses for structured objects?"]

    checkpoints = _std_checkpoints(turns, constraint_text, [kw_polars, kw_dc], tests)
    cp_turns = sorted(set(c["turn"] for c in checkpoints))

    out = {
        "conversation_id": "chatgpt_software_engineering_03",
        "domain": "software_engineering", "domain_name": "Software Engineering",
        "source": "chatgpt", "constraints": [constraint_text],
        "num_turns": num_user, "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints, "turns": turns,
    }
    src.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ SE3 normalised  ({len(turns)} entries, {num_user} user turns, "
          f"checkpoints at turns {cp_turns})")


# ── SE4: Auth / RBAC ──────────────────────────────────────────────────────────

def normalize_se4():
    src = BASE / "chatgpt_software_engineering_04.json"
    data = json.loads(src.read_text(encoding="utf-8"))

    raw = data.get("turns") or data.get("conversation") or []
    turns = _fix_turns(raw)
    num_user = len([t for t in turns if t["role"] == "user"])

    constraint_text = (
        "# Security: comment above every security-sensitive function; "
        "explicit HTTP status codes in all API endpoints (e.g. status_code=200, HTTPException(status_code=403))"
    )
    kw_sec = ["# Security:", "# security:", "# Auth:", "# auth:"]
    kw_status = ["status_code=200", "status_code=201", "status_code=400", "status_code=401",
                 "status_code=403", "status_code=404", "HTTPException(status_code",
                 "status_code=422", "status_code=409"]
    tests = ["Does response include '# Security:' comment above security-sensitive functions?",
             "Does response include explicit HTTP status codes?"]

    checkpoints = _std_checkpoints(turns, constraint_text, [kw_sec, kw_status], tests)
    cp_turns = sorted(set(c["turn"] for c in checkpoints))

    out = {
        "conversation_id": "chatgpt_software_engineering_04",
        "domain": "software_engineering", "domain_name": "Software Engineering",
        "source": "chatgpt", "constraints": [constraint_text],
        "num_turns": num_user, "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints, "turns": turns,
    }
    src.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ SE4 normalised  ({len(turns)} entries, {num_user} user turns, "
          f"checkpoints at turns {cp_turns})")


# ── SE5: CLI tool ──────────────────────────────────────────────────────────────

def normalize_se5():
    src = BASE / "chatgpt_software_engineering_05.json"
    data = json.loads(src.read_text(encoding="utf-8"))

    data["conversation_id"] = "chatgpt_software_engineering_05"

    raw = data.get("turns") or data.get("conversation") or []
    raw_turn_nums = [t.get("turn", 0) for t in raw]
    is_paired = len(raw_turn_nums) != len(set(raw_turn_nums))

    fixed = _fix_turns(raw)
    data["turns"] = fixed
    num_user = len([t for t in fixed if t["role"] == "user"])

    constraint_text = data.get("constraints", [""])[0]
    kw_logging = ["logging", "getLogger", "logger.", "log.", ".info(", ".debug(", ".warning(", ".error("]
    kw_exit = ["sys.exit(0)", "sys.exit(1)", "exit_code = 0", "exit_code = 1", "return 0", "return 1", "sys.exit"]
    tests = ["Does response use logging module instead of print statements?",
             "Does response use consistent exit codes (sys.exit(0) or sys.exit(1))?"]

    checkpoints = _std_checkpoints(fixed, constraint_text, [kw_logging, kw_exit], tests)
    cp_turns = sorted(set(c["turn"] for c in checkpoints))

    data["checkpoints"] = checkpoints
    data["num_turns"] = num_user
    data["num_checkpoints"] = len(checkpoints)

    src.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    paired_note = " (was pair-numbered)" if is_paired else ""
    print(f"✅ SE5 normalised{paired_note}  ({len(fixed)} entries, {num_user} user turns, "
          f"checkpoints at turns {cp_turns})")


# ── Finance_02: India personal finance ────────────────────────────────────────

def normalize_finance02():
    src = BASE / "chatgpt_personal_finance_02.json"
    if not src.exists():
        print("⏭  Finance_02 not found — skipping")
        return

    data = json.loads(src.read_text(encoding="utf-8"))
    raw = data.get("turns") or data.get("conversation") or []
    turns = _fix_turns(raw)
    num_user = len([t for t in turns if t["role"] == "user"])

    constraint_text = (
        "all monetary amounts in ₹ (rupees); only SEBI/AMFI-registered investment products; "
        "no US-centric advice (401k, IRA, S&P 500)"
    )
    kw_inr = ["₹", "Rs.", "rupee", "lakh", "crore", "INR"]
    kw_sebi = ["SEBI", "AMFI", "ELSS", "PPF", "NPS", "mutual fund", "SIP", "FD", "NSE", "BSE", "RD"]
    tests = ["Does response use ₹ or rupee amounts?",
             "Does response mention SEBI/AMFI-registered or India-specific instruments?"]

    checkpoints = _std_checkpoints(turns, constraint_text, [kw_inr, kw_sebi], tests)
    cp_turns = sorted(set(c["turn"] for c in checkpoints))

    out = {
        "conversation_id": "chatgpt_personal_finance_02",
        "domain": "personal_finance", "domain_name": "Personal Finance",
        "source": "chatgpt", "constraints": [constraint_text],
        "num_turns": num_user, "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints, "turns": turns,
    }
    src.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Finance_02 normalised  ({len(turns)} entries, {num_user} user turns, "
          f"checkpoints at turns {cp_turns})")


# ── Validation ────────────────────────────────────────────────────────────────

def validate(filename: str):
    path = BASE / filename
    if not path.exists():
        print(f"  {filename}: NOT FOUND — skipping")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "turns" in data, "missing 'turns'"
    assert "checkpoints" in data, "missing 'checkpoints'"
    assert isinstance(data["checkpoints"], list), "checkpoints must be a list"
    assert all("test" in c and "keywords" in c and "turn" in c
               for c in data["checkpoints"]), "checkpoint missing required fields"
    user_turns = [t for t in data["turns"] if t.get("role") == "user"]
    roles = [t.get("role") for t in data["turns"]]
    consecutive = any(roles[i] == roles[i+1] for i in range(len(roles)-1))
    print(f"  {filename}: {len(data['turns'])} entries, {len(user_turns)} user turns, "
          f"{len(data['checkpoints'])} checkpoints"
          + (" ⚠ consecutive same-role turns!" if consecutive else " — OK"))


if __name__ == "__main__":
    normalize_se3()
    normalize_se4()
    normalize_se5()
    normalize_finance02()

    print("\nValidating...")
    for f in ["chatgpt_software_engineering_03.json",
              "chatgpt_software_engineering_04.json",
              "chatgpt_software_engineering_05.json",
              "chatgpt_personal_finance_02.json"]:
        validate(f)

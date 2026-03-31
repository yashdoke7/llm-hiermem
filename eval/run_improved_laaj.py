#!/usr/bin/env python3
"""
Run a compact multi-system LAAJ pass with an improved prompt.

Writes, per run directory:
- laaj_improved_input.json
- laaj_improved_raw.txt
- laaj_improved.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CHECKPOINT_TURNS = [19, 39, 59, 79, 99]


def _clip(text: Any, max_chars: int) -> str:
    s = "" if text is None else str(text)
    if len(s) <= max_chars:
        return s
    head = max_chars - 180
    return s[:head] + "\n...[truncated for compact judging]...\n" + s[-140:]


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data[0] if data else {}
    return data


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("\njson", "")

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError("Model output did not contain a parseable JSON object.")


def _build_identity_map(
    existing_laaj: Dict[str, Any], systems_in_results: List[str]
) -> Dict[str, str]:
    reveal = existing_laaj.get("system_identity_reveal", {})
    mapped = {k: v for k, v in reveal.items() if k != "ranked_reveal" and isinstance(v, str)}
    if mapped:
        return mapped

    ordered = sorted(systems_in_results)
    return {f"system_{chr(ord('a') + i)}": name for i, name in enumerate(ordered)}


def _compact_transcript(turn_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    constraint_turns = {
        int(t.get("turn"))
        for t in turn_logs
        if "constraint" in str(t.get("type", "")).lower()
    }
    keep_turns = set(CHECKPOINT_TURNS) | constraint_turns | {1}

    out: List[Dict[str, Any]] = []
    for t in sorted(turn_logs, key=lambda x: int(x.get("turn", 0))):
        turn = int(t.get("turn", 0))
        if turn not in keep_turns:
            continue

        item: Dict[str, Any] = {
            "turn": turn,
            "type": t.get("type"),
        }
        user = t.get("user")
        if user is not None:
            if turn in CHECKPOINT_TURNS:
                item["user"] = _clip(user, 550)
            elif turn == 1:
                item["user"] = _clip(user, 380)
            else:
                item["user"] = _clip(user, 500)

        assistant = t.get("assistant")
        if assistant is None:
            assistant = t.get("response")

        # Keep assistant text at checkpoints (plus turn 1 for initial context).
        if turn in CHECKPOINT_TURNS or turn == 1:
            limit = 1250 if turn in CHECKPOINT_TURNS else 450
            item["assistant"] = _clip(assistant or "", limit)

        out.append(item)
    return out


def _build_payload(run_dir: Path) -> Dict[str, Any]:
    results_path = run_dir / "results.json"
    laaj_path = run_dir / "laaj.json"
    results = _load_json(results_path)
    existing_laaj = _load_json(laaj_path) if laaj_path.exists() else {}

    systems = [s for s in ["hiermem", "rag", "raw_llm", "rag_summary"] if s in results]
    identity_map = _build_identity_map(existing_laaj, systems)

    transcripts: Dict[str, List[Dict[str, Any]]] = {}
    for anon_key, real_name in identity_map.items():
        system_rows = results.get(real_name, [])
        if isinstance(system_rows, list) and system_rows:
            record = system_rows[0]
        elif isinstance(system_rows, dict):
            record = system_rows
        else:
            transcripts[anon_key] = []
            continue
        transcripts[anon_key] = _compact_transcript(record.get("turn_logs", []))

    old_meta = existing_laaj.get("evaluation_metadata", {})
    first_system = systems[0] if systems else None
    first_record = {}
    if first_system and results.get(first_system):
        if isinstance(results[first_system], list) and results[first_system]:
            first_record = results[first_system][0]
        elif isinstance(results[first_system], dict):
            first_record = results[first_system]

    dataset_id = (
        old_meta.get("dataset_id")
        or old_meta.get("benchmark")
        or first_record.get("conversation_id")
        or run_dir.name
    )

    payload = {
        "evaluation_metadata": {
            "dataset_id": dataset_id,
            "domain": old_meta.get("domain", "software_engineering"),
            "topic": old_meta.get("topic", "constraint_tracking"),
            "total_conversation_turns": old_meta.get("total_conversation_turns", 100),
            "systems_evaluated": list(identity_map.keys()),
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "constraint_registry": existing_laaj.get("constraint_registry", []),
        "checkpoint_turns_expected": CHECKPOINT_TURNS,
        "turn_indexing_mode": "legacy_odd_raw",
        "system_transcripts": transcripts,
        "SYSTEM_KEY_MAP_REVEAL": identity_map,
    }
    return payload


def _ranking_from_laaj(laaj_obj: Dict[str, Any]) -> List[Tuple[int, str, float]]:
    rankings = laaj_obj.get("cross_system_analysis", {}).get("final_rankings", {})
    reveal = laaj_obj.get("system_identity_reveal", {})
    rows: List[Tuple[int, str, float]] = []

    if isinstance(rankings, dict) and rankings:
        for rk, entry in rankings.items():
            try:
                rank = int(rk)
            except Exception:
                continue
            anon = entry.get("system")
            score = float(entry.get("average_score", 0.0))
            name = reveal.get(anon, anon)
            rows.append((rank, name, score))
        rows.sort(key=lambda x: x[0])
        return rows

    # Fallback: derive from system_evaluations
    sys_eval = laaj_obj.get("system_evaluations", {})
    tmp: List[Tuple[str, float]] = []
    for anon, ev in sys_eval.items():
        avg = ev.get("average_score")
        if avg is None:
            cps = ev.get("checkpoints", [])
            vals = [float(c.get("weighted_score", 0.0)) for c in cps if "weighted_score" in c]
            avg = sum(vals) / len(vals) if vals else 0.0
        tmp.append((reveal.get(anon, anon), float(avg)))
    tmp.sort(key=lambda x: x[1], reverse=True)
    return [(i + 1, n, s) for i, (n, s) in enumerate(tmp)]


def _ranking_from_metrics_fixed(metrics_path: Path) -> List[Tuple[int, str, float]]:
    obj = _load_json(metrics_path)
    vals: List[Tuple[str, float]] = []
    for system, row in obj.items():
        if not isinstance(row, dict):
            continue
        judge_avg = row.get("judge_scores", {}).get("average_score")
        if isinstance(judge_avg, (int, float)):
            score = float(judge_avg)
        else:
            pts = row.get("per_turn_scores", {})
            nums = [float(v) for v in pts.values() if isinstance(v, (int, float))]
            score = sum(nums) / len(nums) if nums else 0.0
        vals.append((system, score))
    vals.sort(key=lambda x: x[1], reverse=True)
    return [(i + 1, n, s) for i, (n, s) in enumerate(vals)]


def run_once(
    run_dir: Path,
    prompt_text: str,
    provider: str,
    model: str,
    max_tokens: int,
) -> None:
    payload = _build_payload(run_dir)
    input_path = run_dir / "laaj_improved_input.json"
    raw_path = run_dir / "laaj_improved_raw.txt"
    out_path = run_dir / "laaj_improved.json"

    input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Delayed import so OLLAMA_CONTEXT_SIZE env override is honored.
    from llm.client import LLMClient

    client = LLMClient(provider=provider)
    user_prompt = (
        "Evaluate the payload below and return JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    text = client.call(
        system_prompt=prompt_text,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    raw_path.write_text(text, encoding="utf-8")
    parsed = _extract_json_object(text)

    # Ensure identity reveal exists for downstream compatibility.
    if "system_identity_reveal" not in parsed:
        parsed["system_identity_reveal"] = payload["SYSTEM_KEY_MAP_REVEAL"]
    else:
        for k, v in payload["SYSTEM_KEY_MAP_REVEAL"].items():
            parsed["system_identity_reveal"].setdefault(k, v)

    out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    improved_rank = _ranking_from_laaj(parsed)
    metrics_rank = _ranking_from_metrics_fixed(run_dir / "metrics_rescored_fixed.json")

    print(f"\n=== {run_dir} ===")
    print("improved_laaaj:", " > ".join([f"{r}:{n}({s:.2f})" for r, n, s in improved_rank]))
    print("metrics_fixed :", " > ".join([f"{r}:{n}({s:.2f})" for r, n, s in metrics_rank]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run improved compact LAAJ for one or more dataset folders.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Dataset run directories containing results.json and laaj.json.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt_improved_compact.txt",
        help="Path to improved judge prompt file.",
    )
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--num-ctx", type=int, default=32768)
    args = parser.parse_args()

    os.environ["OLLAMA_CONTEXT_SIZE"] = str(args.num_ctx)

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    for rd in args.run_dirs:
        run_once(
            run_dir=Path(rd),
            prompt_text=prompt_text,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()

"""
normalize_datasets.py

Normalize all dataset_*_se.json files under:
  eval/datasets/constraint_tracking/llm_generated/

Target schema is aligned to dataset_01_se.json.

Run:
  python -m eval.normalize_datasets
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE = Path(__file__).parent / "datasets" / "constraint_tracking" / "llm_generated"


@dataclass
class NormalizeReport:
    file: str
    changed: bool
    warnings: List[str]


def _constraint_base_id(cid: str) -> str:
    cid = (cid or "").strip()
    cid = re.sub(r"_prime$", "", cid, flags=re.IGNORECASE)
    cid = re.sub(r"\bprime\b$", "", cid, flags=re.IGNORECASE).strip("_")
    return cid or "C_UNKNOWN"


def _normalize_turns(turns: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    out: List[Dict[str, Any]] = []

    for i, t in enumerate(turns, start=1):
        role = t.get("role") or t.get("speaker")
        if role not in ("user", "assistant"):
            role = "user" if i % 2 == 1 else "assistant"
            warnings.append(f"turn {i}: invalid/missing role -> inferred {role}")

        text = t.get("text")
        if text is None:
            text = t.get("content", "")

        # dataset_01 style: user keeps topic/constraint_* labels; assistant uses response
        old_type = t.get("type")
        if role == "assistant":
            new_type = "response"
            if old_type and old_type != "response":
                warnings.append(f"turn {i}: assistant type '{old_type}' -> 'response'")
        else:
            allowed_user = {
                "topic",
                "clarification",
                "constraint_add",
                "constraint_modify",
                "constraint_remove",
                "constraint_stack",
                "back_reference",
                "pushback",
            }
            new_type = old_type if old_type in allowed_user else "topic"
            if old_type and new_type != old_type:
                warnings.append(f"turn {i}: user type '{old_type}' -> 'topic'")

        out.append({
            "turn": i,
            "role": role,
            "text": text,
            "type": new_type,
        })

    # Detect but do not rewrite conversational semantics if alternation is broken.
    for i in range(1, len(out)):
        if out[i]["role"] == out[i - 1]["role"]:
            warnings.append(
                "found consecutive same-role turns; kept order to avoid semantic corruption"
            )
            break

    return out, warnings


def _normalize_constraint_log(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    warnings: List[str] = []
    cl = data.get("constraint_log", [])

    # Already close to target schema.
    if cl and "id" in cl[0]:
        id_to_text: Dict[str, str] = {}
        out = []
        for row in cl:
            cid = row.get("id")
            out.append({
                "id": cid,
                "introduced_turn": row.get("introduced_turn"),
                "removed_turn": row.get("removed_turn"),
                "status_at_end": row.get("status_at_end"),
                "original_text": row.get("original_text", ""),
                "final_text": row.get("final_text", row.get("original_text", "")),
                "lifecycle": row.get("lifecycle", []),
            })
            id_to_text[cid] = out[-1]["final_text"]
        return out, id_to_text, warnings

    # Event style -> compact style.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ev in cl:
        base = _constraint_base_id(ev.get("constraint_id", ""))
        grouped.setdefault(base, []).append(ev)

    out: List[Dict[str, Any]] = []
    id_to_text: Dict[str, str] = {}

    for cid in sorted(grouped.keys()):
        evs = sorted(grouped[cid], key=lambda e: e.get("event_turn", 10**9))
        introduced_turn = None
        removed_turn = None
        lifecycle: List[str] = []
        original_text = ""
        final_text = ""
        modified = False

        for ev in evs:
            et = ev.get("event_type")
            t = ev.get("event_turn")
            desc = ev.get("description", "")

            if et in ("constraint_add", "constraint_stack") and introduced_turn is None:
                introduced_turn = t
                original_text = desc or original_text
                final_text = desc or final_text
                lifecycle.append(f"introduced@{t}")
            elif et == "constraint_modify":
                if introduced_turn is None:
                    introduced_turn = t
                    lifecycle.append(f"introduced@{t}")
                    original_text = desc or original_text
                lifecycle.append(f"modified@{t}")
                final_text = desc or final_text
                modified = True
            elif et == "constraint_remove":
                removed_turn = t
                lifecycle.append(f"removed@{t}")
            else:
                warnings.append(f"constraint {cid}: unknown event_type '{et}'")

            if not final_text and desc:
                final_text = desc

        if introduced_turn is None:
            warnings.append(f"constraint {cid}: no add/stack event; inferred introduced_turn from first event")
            introduced_turn = evs[0].get("event_turn") if evs else None

        status = "removed" if removed_turn is not None else ("active_modified" if modified else "active")
        if not original_text:
            original_text = final_text
        if not final_text:
            final_text = original_text

        row = {
            "id": cid,
            "introduced_turn": introduced_turn,
            "removed_turn": removed_turn,
            "status_at_end": status,
            "original_text": original_text,
            "final_text": final_text,
            "lifecycle": lifecycle,
        }
        out.append(row)
        id_to_text[cid] = final_text or original_text

    out.sort(key=lambda r: (r.get("introduced_turn") if r.get("introduced_turn") is not None else 10**9, r["id"]))
    return out, id_to_text, warnings


def _keywords_from_text(text: str, test_q: str) -> List[str]:
    s = f"{text} {test_q}".lower()
    kws: List[str] = []
    if "mlflow" in s:
        kws.extend(["mlflow", "log_param", "set_tag"])
    if "latency" in s:
        kws.append("latency_ms")
    if "model_version" in s:
        kws.append("model_version")
    if "tracer.start_as_current_span" in s or "span" in s:
        kws.append("tracer.start_as_current_span")
    if "service.operation" in s:
        kws.append("service.operation")
    if "session.begin" in s:
        kws.append("async with session.begin()")
    if "taskid" in s or "# taskid" in s:
        kws.append("# TaskID:")
    if "permissions" in s and "contents: read" in s:
        kws.extend(["permissions", "contents: read"])
    if "trivy" in s:
        kws.append("trivy-scan")
    if "soc2" in s:
        kws.append("Compliance")
    if not kws:
        tokens = [w.strip('.,:;()[]{}"\'') for w in (text or "").split()]
        kws = [t for t in tokens if len(t) > 4][:3] or ["constraint"]
    # Preserve order, de-dup
    seen = set()
    out = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _normalize_checkpoints(
    checkpoints: List[Dict[str, Any]],
    id_to_text: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    out: List[Dict[str, Any]] = []

    for cp in checkpoints:
        cp_turn = cp.get("checkpoint_turn", cp.get("turn"))
        active = cp.get("active_constraints") or []
        tests = cp.get("tests") or []

        norm_tests: List[Dict[str, Any]] = []
        for t in tests:
            cid_raw = t.get("constraint_id", "")
            cid = _constraint_base_id(cid_raw)
            ctext = (
                t.get("constraint_text")
                or id_to_text.get(cid)
                or id_to_text.get(cid_raw)
                or ""
            )
            tq = t.get("test_question") or t.get("question") or ""
            ans_raw = t.get("answer", True)
            if ans_raw is None:
                ans = None
            else:
                ans = bool(ans_raw)
            kws = t.get("keywords")
            if not isinstance(kws, list) or not kws:
                kws = _keywords_from_text(ctext, tq)

            norm_tests.append({
                "constraint_id": cid,
                "constraint_text": ctext,
                "test_question": tq,
                "answer": ans,
                "keywords": kws,
            })

        if not active and norm_tests:
            # Derive active constraints from tests if missing.
            active = sorted({t["constraint_id"] for t in norm_tests})
            warnings.append(f"checkpoint {cp_turn}: active_constraints missing -> derived from tests")

        out.append({
            "checkpoint_turn": cp_turn,
            "active_constraints": active,
            "tests": norm_tests,
        })

    out.sort(key=lambda c: c.get("checkpoint_turn", 10**9))
    return out, warnings


def normalize_to_dataset01_schema(path: Path) -> NormalizeReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    warnings: List[str] = []

    # Top-level scalar defaults
    conv_id = data.get("conversation_id") or path.stem
    domain = data.get("domain", "software_engineering")
    domain_name = data.get("domain_name", "Software Engineering")
    topic = data.get("topic", "")
    source = data.get("source") or "GPT-5.3-Codex"

    turns_raw = data.get("turns") or []
    turns, turn_warn = _normalize_turns(turns_raw)
    warnings.extend(turn_warn)

    cl_norm, id_to_text, cl_warn = _normalize_constraint_log(data)
    warnings.extend(cl_warn)

    cps_raw = data.get("checkpoints") or []
    cps_norm, cp_warn = _normalize_checkpoints(cps_raw, id_to_text)
    warnings.extend(cp_warn)

    num_user_turns = len([t for t in turns if t["role"] == "user"])
    num_checkpoints = len(cps_norm)

    normalized = {
        "conversation_id": conv_id,
        "domain": domain,
        "domain_name": domain_name,
        "topic": topic,
        "source": source,
        "constraint_log": cl_norm,
        "num_turns": num_user_turns,
        "num_checkpoints": num_checkpoints,
        "checkpoints": cps_norm,
        "turns": turns,
    }

    old_serialized = json.dumps(data, ensure_ascii=False, sort_keys=True)
    new_serialized = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    changed = old_serialized != new_serialized

    if changed:
        path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    # Non-fixable semantic checks
    cp_false = 0
    for cp in cps_norm:
        for t in cp.get("tests", []):
            if t.get("answer") is False:
                cp_false += 1
    if cp_false == 0:
        warnings.append("no false checkpoint answers present; judge sensitivity may be weak")

    return NormalizeReport(file=path.name, changed=changed, warnings=warnings)


def _validate_shape(path: Path) -> List[str]:
    issues: List[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = data.get("turns", [])
    cps = data.get("checkpoints", [])

    if data.get("num_turns") != len([t for t in turns if t.get("role") == "user"]):
        issues.append("num_turns does not match number of user turns")
    if data.get("num_checkpoints") != len(cps):
        issues.append("num_checkpoints does not match checkpoint array length")

    for i, t in enumerate(turns, start=1):
        if t.get("turn") != i:
            issues.append("turn numbering is not sequential")
            break

    for i in range(1, len(turns)):
        if turns[i - 1].get("role") == turns[i].get("role"):
            issues.append("consecutive same-role turns remain (manual review may be needed)")
            break

    required_cp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cp_turns = [cp.get("checkpoint_turn") for cp in cps]
    if cp_turns != required_cp:
        issues.append("checkpoint turns differ from expected 10-step grid")

    return issues


def main() -> None:
    files = sorted(BASE.glob("dataset_*_se.json"))
    if not files:
        print("No dataset_*_se.json files found.")
        return

    print(f"Normalizing {len(files)} SE dataset files to dataset_01 schema...\n")
    reports: List[NormalizeReport] = []
    for path in files:
        report = normalize_to_dataset01_schema(path)
        reports.append(report)
        status = "UPDATED" if report.changed else "UNCHANGED"
        print(f"- {path.name}: {status}")
        for w in report.warnings:
            print(f"    warning: {w}")

    print("\nValidation summary:")
    for path in files:
        issues = _validate_shape(path)
        if issues:
            print(f"- {path.name}: ISSUES")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"- {path.name}: OK")


if __name__ == "__main__":
    main()

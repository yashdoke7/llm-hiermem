"""
Paper Metrics — Extracts research-grade metrics from existing results.json.

Metrics computed (no re-running required):
  1.  Judge score trend          — avg judge score at early/mid/late turns
  2.  First violation turn        — turn where constraint first fails per system/convo
  3.  Constraint retrieval rate   — % HierMem turns where constraint zone was active
  4.  Auto-correction rate        — % turns HierMem triggered violation_retry
  5.  Response length stability   — std dev of response length across turns (consistency)
  6.  Context utilization         — avg % of configured token budget actually used
  7.  Latency trend               — avg latency at early/mid/late turns
  8.  HierMem mode transition      — turn at which passthrough→HYBRID switches
  9.  Degradation slope           — how fast judge score drops (slope T8→T25)
  10. CVR per convo per system

Usage:
  python -m eval.paper_metrics results/raw/benchmarks/qwen14b_run_v2
  python -m eval.paper_metrics results/raw/benchmarks/qwen14b_run_v2 --json
"""

import json
import sys
import argparse
import math
from pathlib import Path
from statistics import mean, stdev
from typing import List, Dict, Any, Optional

import config

CONTEXT_BUDGET = config.TOTAL_CONTEXT_BUDGET


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_turn_logs(conv: dict) -> list:
    return conv.get("turn_logs", conv.get("results", []))


def _get_checkpoints(conv: dict) -> list:
    return conv.get("checkpoints", [])


def _response_at(turn_logs: list, turn: int) -> Optional[str]:
    for t in turn_logs:
        if t.get("turn") == turn:
            r = t.get("response", "")
            return r if r and not r.startswith("ERROR:") else None
    return None


def _pipeline_at(turn_logs: list, turn: int) -> dict:
    for t in turn_logs:
        if t.get("turn") == turn:
            return t.get("pipeline_details") or {}
    return {}


def _is_passthrough(turn_log: dict) -> bool:
    pd = turn_log.get("pipeline_details") or {}
    warnings = pd.get("warnings", [])
    sources = pd.get("sources_used", [])
    return (any("passthrough" in str(w) for w in warnings) or
            any("passthrough" in str(s) for s in sources))


def _constraint_active(turn_log: dict) -> bool:
    """True if HierMem injected constraint zone content this turn."""
    pd = turn_log.get("pipeline_details") or {}
    zb = pd.get("zone_breakdown", {})
    return zb.get("constraints", 0) > 0


def _violation_retry_count(turn_log: dict) -> int:
    pd = turn_log.get("pipeline_details") or {}
    for w in pd.get("warnings", []):
        if "violation_retry" in str(w):
            import re
            m = re.search(r'(\d+)\s*violations', str(w))
            return int(m.group(1)) if m else 1
    return 0


def _context_tokens(turn_log: dict) -> int:
    pd = turn_log.get("pipeline_details") or {}
    ctx = pd.get("context_tokens_used", 0)
    if not ctx:
        ctx = turn_log.get("context_tokens", 0)
    return ctx


def _keyword_pass(response: str, keywords: list) -> bool:
    if not keywords or not response:
        return False
    rl = response.lower()
    return any(kw.lower() in rl for kw in keywords)


# ─── Metric Extractors ────────────────────────────────────────────────────────

def first_violation_turn(system_results: list) -> dict:
    """Per-conversation: turn at which first constraint violation occurs."""
    out = {}
    for conv in system_results:
        cid = conv.get("conversation_id", "?")
        turn_logs = _get_turn_logs(conv)
        checkpoints = sorted(_get_checkpoints(conv), key=lambda c: c.get("turn", 0))

        first_fail = None
        for cp in checkpoints:
            turn = cp.get("turn", 0)
            response = _response_at(turn_logs, turn)
            if response is None:
                continue
            kws = cp.get("keywords", [])
            if kws and not _keyword_pass(response, kws):
                first_fail = turn
                break
        out[cid] = first_fail  # None = never violated within dataset
    return out


def judge_score_trend(system_results: list, metrics_detail: list) -> dict:
    """Avg judge score at early (≤10), mid (11-20), late (>20) turns."""
    # metrics_detail = list of {turn, convo, judge_score} from metrics_rescored.json
    buckets = {"early": [], "mid": [], "late": []}
    for entry in metrics_detail:
        turn = entry.get("turn", 0)
        score = entry.get("judge_score", 0)
        if turn <= 10:
            buckets["early"].append(score)
        elif turn <= 20:
            buckets["mid"].append(score)
        else:
            buckets["late"].append(score)
    return {k: round(mean(v), 2) if v else None for k, v in buckets.items()}


def degradation_slope(judge_detail: list) -> float:
    """Linear slope of judge score over turns (negative = degrading).
    
    Uses least-squares fit over all (turn, score) pairs.
    """
    points = [(d.get("turn", 0), d.get("judge_score", 0)) for d in judge_detail]
    if len(points) < 2:
        return 0.0
    n = len(points)
    turns = [p[0] for p in points]
    scores = [p[1] for p in points]
    t_mean = mean(turns)
    s_mean = mean(scores)
    num = sum((t - t_mean) * (s - s_mean) for t, s in zip(turns, scores))
    den = sum((t - t_mean) ** 2 for t in turns)
    return round(num / den, 4) if den != 0 else 0.0


def constraint_retrieval_rate(system_results: list, system_name: str) -> Optional[float]:
    """HierMem only: % of turns (after passthrough phase) where constraint was in context."""
    if system_name != "hiermem":
        return None
    total, active = 0, 0
    for conv in system_results:
        for t in _get_turn_logs(conv):
            if not _is_passthrough(t):
                total += 1
                if _constraint_active(t):
                    active += 1
    return round(active / total, 4) if total > 0 else None


def auto_correction_stats(system_results: list, system_name: str) -> Optional[dict]:
    """HierMem only: turns that triggered violation_retry, avg corrections."""
    if system_name != "hiermem":
        return None
    triggered, total_hybrid, total_corrections = 0, 0, 0
    for conv in system_results:
        for t in _get_turn_logs(conv):
            if not _is_passthrough(t):
                total_hybrid += 1
                n = _violation_retry_count(t)
                if n > 0:
                    triggered += 1
                    total_corrections += n
    if total_hybrid == 0:
        return None
    return {
        "hybrid_turns": total_hybrid,
        "triggered_turns": triggered,
        "trigger_rate": round(triggered / total_hybrid, 4),
        "total_corrections": total_corrections,
    }


def response_length_stats(system_results: list) -> dict:
    """Avg and std dev of response length (chars) across all turns."""
    lengths = []
    for conv in system_results:
        for t in _get_turn_logs(conv):
            r = t.get("response", "")
            if r and not r.startswith("ERROR:"):
                lengths.append(len(r))
    if not lengths:
        return {}
    return {
        "avg_chars": round(mean(lengths)),
        "std_chars": round(stdev(lengths)) if len(lengths) > 1 else 0,
        "cv": round(stdev(lengths) / mean(lengths), 3) if len(lengths) > 1 else 0,
        # CV (coefficient of variation) — lower = more consistent output length
    }


def context_utilization(system_results: list) -> dict:
    """Avg % of CONTEXT_BUDGET actually used per turn."""
    usages = []
    for conv in system_results:
        for t in _get_turn_logs(conv):
            ctx = _context_tokens(t)
            if ctx > 0:
                usages.append(ctx / CONTEXT_BUDGET)
    if not usages:
        return {}
    return {
        "avg_utilization_pct": round(mean(usages) * 100, 1),
        "max_utilization_pct": round(max(usages) * 100, 1),
        "avg_tokens": round(mean(usages) * CONTEXT_BUDGET),
    }


def latency_trend(system_results: list) -> dict:
    """Avg latency at early/mid/late turns."""
    buckets = {"early": [], "mid": [], "late": []}
    for conv in system_results:
        for t in _get_turn_logs(conv):
            turn = t.get("turn", 0)
            lat = t.get("latency_seconds", 0)
            if not lat:
                continue
            if turn <= 10:
                buckets["early"].append(lat)
            elif turn <= 20:
                buckets["mid"].append(lat)
            else:
                buckets["late"].append(lat)
    return {k: round(mean(v), 1) if v else None for k, v in buckets.items()}


def mode_transition_turn(system_results: list, system_name: str) -> Optional[int]:
    """HierMem only: first turn where HYBRID mode is used (end of passthrough phase)."""
    if system_name != "hiermem":
        return None
    earliest = None
    for conv in system_results:
        for t in _get_turn_logs(conv):
            pd = t.get("pipeline_details") or {}
            if pd.get("curator_strategy") == "HYBRID":
                turn = t.get("turn", 0)
                if earliest is None or turn < earliest:
                    earliest = turn
    return earliest


def cvr_per_convo(system_results: list) -> dict:
    """Constraint violation rate per conversation."""
    out = {}
    for conv in system_results:
        cid = conv.get("conversation_id", "?")
        turn_logs = _get_turn_logs(conv)
        checkpoints = _get_checkpoints(conv)
        total, violations = 0, 0
        seen_turns = set()
        for cp in checkpoints:
            turn = cp.get("turn", 0)
            if turn in seen_turns:
                continue
            seen_turns.add(turn)
            response = _response_at(turn_logs, turn)
            if response is None:
                continue
            total += 1
            kws = cp.get("keywords", [])
            if kws and not _keyword_pass(response, kws):
                violations += 1
        out[cid] = round(violations / total, 4) if total > 0 else 0.0
    return out


def memory_efficiency(all_results: dict) -> Optional[dict]:
    """Ratio of HierMem token usage to raw_llm token usage per turn.
    
    HierMem's key claim: same (or better) constraint adherence at lower context cost.
    efficiency_ratio < 1.0 means HierMem used fewer tokens than raw_llm.
    """
    if "hiermem" not in all_results or "raw_llm" not in all_results:
        return None
    
    # Collect per-turn token usage for both systems
    hiermem_tokens = []
    raw_llm_tokens = []
    
    hiermem_convos = {c.get("conversation_id"): c for c in all_results["hiermem"]}
    raw_llm_convos = {c.get("conversation_id"): c for c in all_results["raw_llm"]}
    
    for cid in hiermem_convos:
        if cid not in raw_llm_convos:
            continue
        for t in _get_turn_logs(hiermem_convos[cid]):
            ctx = _context_tokens(t)
            if ctx > 0:
                hiermem_tokens.append(ctx)
        for t in _get_turn_logs(raw_llm_convos[cid]):
            ctx = _context_tokens(t)
            if ctx > 0:
                raw_llm_tokens.append(ctx)
    
    if not hiermem_tokens or not raw_llm_tokens:
        return None
    
    avg_hiermem = mean(hiermem_tokens)
    avg_raw = mean(raw_llm_tokens)
    return {
        "avg_tokens_hiermem": round(avg_hiermem),
        "avg_tokens_raw_llm": round(avg_raw),
        "efficiency_ratio": round(avg_hiermem / avg_raw, 3) if avg_raw > 0 else None,
        "tokens_saved_pct": round((1 - avg_hiermem / avg_raw) * 100, 1) if avg_raw > 0 else None,
    }


def per_domain_accuracy(all_results: dict, metrics_rescored: dict) -> dict:
    """Per-domain breakdown of judge score and CVR.
    
    Splits results by conversation domain (software_engineering, personal_finance, etc.)
    so we can show whether HierMem advantage is domain-specific.
    """
    # Build domain map from conversation metadata in results
    domain_map = {}  # cid -> domain
    for system_results in all_results.values():
        for conv in system_results:
            cid = conv.get("conversation_id", "?")
            # Try to infer domain from conversation_id
            if "software_engineering" in cid or "_se" in cid:
                domain_map[cid] = "software_engineering"
            elif "personal_finance" in cid or "finance" in cid:
                domain_map[cid] = "personal_finance"
            elif "cooking" in cid or "recipe" in cid:
                domain_map[cid] = "cooking"
            else:
                domain_map[cid] = "other"

    output = {}
    for system_name, system_results in all_results.items():
        judge_detail = (metrics_rescored.get(system_name, {})
                        .get("judge_scores", {}).get("detail", []))

        # Group judge scores by domain
        domain_scores = {}
        for entry in judge_detail:
            cid = entry.get("convo", "?")
            domain = domain_map.get(cid, "other")
            domain_scores.setdefault(domain, []).append(entry.get("judge_score", 0))

        output[system_name] = {
            domain: round(mean(scores), 2)
            for domain, scores in domain_scores.items()
            if scores
        }
    return output


def degradation_by_position(all_results: dict, metrics_rescored: dict) -> dict:
    """Constraint adherence score at each normalized position (20/40/60/80/100%).
    
    Normalizes checkpoint turns to position labels so plots are comparable
    across datasets with different lengths (50-turn SE vs 30-turn Finance).
    
    The 5 checkpoints in each dataset represent 20/40/60/80/100% of user turns,
    so position index 0=20%, 1=40%, 2=60%, 3=80%, 4=100%.
    
    Returns {system: {position_pct: avg_judge_score}}.
    """
    POSITIONS = ["20%", "40%", "60%", "80%", "100%"]
    
    output = {}
    for system_name, system_results in all_results.items():
        judge_detail = (metrics_rescored.get(system_name, {})
                        .get("judge_scores", {}).get("detail", []))

        # Build: cid -> sorted list of checkpoint turns in this conversation
        cid_checkpoints = {}
        for conv in system_results:
            cid = conv.get("conversation_id", "?")
            cp_turns = sorted(set(
                cp.get("turn", 0) for cp in conv.get("checkpoints", [])
                if cp.get("turn", 0) > 0
            ))
            if cp_turns:
                cid_checkpoints[cid] = cp_turns

        # Map each judge_detail entry to its position index
        position_scores = {p: [] for p in POSITIONS}
        for entry in judge_detail:
            cid = entry.get("convo", "?")
            turn = entry.get("turn", 0)
            cp_turns = cid_checkpoints.get(cid, [])
            if not cp_turns:
                continue
            try:
                idx = cp_turns.index(turn)
                if idx < len(POSITIONS):
                    position_scores[POSITIONS[idx]].append(entry.get("judge_score", 0))
            except ValueError:
                pass

        output[system_name] = {
            pos: round(mean(scores), 2) if scores else None
            for pos, scores in position_scores.items()
        }
    return output


# ─── NEW: Token-Accuracy Curve ────────────────────────────────────────────────

def token_accuracy_curve(all_results: dict, metrics_rescored: dict) -> dict:
    """Map context tokens used → judge score for every checkpoint turn per system.

    Produces per-system lists of (tokens, score) pairs that can be plotted
    directly as scatter or line charts.  This is the key efficiency chart:
    HierMem should cluster in the low-token / high-score quadrant.

    Returns {system: [{turn, convo, tokens, judge_score}, ...]}.
    """
    output = {}
    for system_name, system_results in all_results.items():
        judge_detail = (metrics_rescored.get(system_name, {})
                        .get("judge_scores", {}).get("detail", []))
        judge_map = {(d["convo"], d["turn"]): d["judge_score"] for d in judge_detail}

        points = []
        for conv in system_results:
            cid = conv.get("conversation_id", "?")
            for tl in _get_turn_logs(conv):
                turn = tl.get("turn", 0)
                ctx = _context_tokens(tl)
                score = judge_map.get((cid, turn))
                if score is not None and ctx > 0:
                    points.append({
                        "turn": turn, "convo": cid,
                        "tokens": ctx, "judge_score": score,
                    })
        output[system_name] = points
    return output


def cost_quality_ratio(all_results: dict, metrics_rescored: dict) -> dict:
    """Compute cost-quality trade-off per system.

    Cost proxy = total estimated tokens consumed across ALL LLM calls per turn
    (context tokens × num_calls).  For hiermem, pipeline_details may contain
    multi-call breakdown; for baselines it's a single call.

    Returns {system: {avg_quality, avg_cost_tokens, cost_per_quality_point,
                      total_turns}}.
    """
    output = {}
    for system_name, system_results in all_results.items():
        judge_detail = (metrics_rescored.get(system_name, {})
                        .get("judge_scores", {}).get("detail", []))
        avg_score = mean([d["judge_score"] for d in judge_detail]) if judge_detail else 0

        total_tokens, total_turns = 0, 0
        for conv in system_results:
            for tl in _get_turn_logs(conv):
                ctx = _context_tokens(tl)
                if ctx > 0:
                    # Estimate call multiplier from pipeline details
                    pd = tl.get("pipeline_details") or {}
                    sources = pd.get("sources_used", [])
                    is_hiermem_full = any("passthrough" not in str(s) for s in sources) and len(sources) > 1
                    # hiermem full pipeline: curator(~1K) + main(~6K) + postproc(~1K) ≈ ctx * 1.3
                    # baselines: 1 call = ctx
                    if system_name == "hiermem" and is_hiermem_full:
                        total_tokens += int(ctx * 1.3)
                    elif system_name == "rag_summary":
                        total_tokens += int(ctx * 1.15)  # main + summary update
                    else:
                        total_tokens += ctx
                    total_turns += 1

        avg_cost = round(total_tokens / max(total_turns, 1))
        cost_per_point = round(avg_cost / max(avg_score, 0.1), 1)

        output[system_name] = {
            "avg_quality": round(avg_score, 2),
            "avg_cost_tokens": avg_cost,
            "cost_per_quality_point": cost_per_point,
            "total_turns": total_turns,
        }
    return output


def constraint_retention_by_distance(all_results: dict, metrics_rescored: dict) -> dict:
    """Measure how constraint score changes as distance from constraint setup grows.

    Groups checkpoints by how many turns have passed since the constraint was
    set (turn 1), then averages judge scores per distance bucket.
    Buckets: 0-10, 11-20, 21-30, 31-40, 41-50 turns from constraint.

    Directly tests the "Lost in the Middle" hypothesis: baselines should
    degrade with distance while HierMem stays flat.

    Returns {system: {bucket_label: avg_judge_score}}.
    """
    BUCKETS = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    LABELS = ["T1-10", "T11-20", "T21-30", "T31-40", "T41-50"]

    output = {}
    for system_name in all_results:
        judge_detail = (metrics_rescored.get(system_name, {})
                        .get("judge_scores", {}).get("detail", []))

        bucket_scores = {label: [] for label in LABELS}
        for d in judge_detail:
            turn = d.get("turn", 0)
            score = d.get("judge_score", 0)
            for (lo, hi), label in zip(BUCKETS, LABELS):
                if lo <= turn <= hi:
                    bucket_scores[label].append(score)
                    break
            else:
                # Turn > 50 — put in last bucket
                if turn > 50:
                    bucket_scores[LABELS[-1]].append(score)

        output[system_name] = {
            label: round(mean(scores), 2) if scores else None
            for label, scores in bucket_scores.items()
        }
    return output


def analyze_all(results: dict, metrics_rescored: dict) -> dict:
    output = {}
    for system_name, system_results in results.items():
        judge_detail = (metrics_rescored.get(system_name, {})
                        .get("judge_scores", {})
                        .get("detail", []))

        output[system_name] = {
            "judge_score_trend": judge_score_trend(system_results, judge_detail),
            "degradation_slope": degradation_slope(judge_detail),
            "first_violation_turn": first_violation_turn(system_results),
            "cvr_per_convo": cvr_per_convo(system_results),
            "response_length_stats": response_length_stats(system_results),
            "context_utilization": context_utilization(system_results),
            "latency_trend": latency_trend(system_results),
            "constraint_retrieval_rate": constraint_retrieval_rate(system_results, system_name),
            "auto_correction_stats": auto_correction_stats(system_results, system_name),
            "mode_transition_turn": mode_transition_turn(system_results, system_name),
        }

    # Cross-system metrics (not per-system)
    output["_cross_system"] = {
        "memory_efficiency": memory_efficiency(results),
        "per_domain_accuracy": per_domain_accuracy(results, metrics_rescored),
        "degradation_by_position": degradation_by_position(results, metrics_rescored),
        "token_accuracy_curve": token_accuracy_curve(results, metrics_rescored),
        "cost_quality_ratio": cost_quality_ratio(results, metrics_rescored),
        "constraint_retention_by_distance": constraint_retention_by_distance(results, metrics_rescored),
    }
    return output


def print_report(data: dict):
    systems = [s for s in data.keys() if not s.startswith("_")]
    print("\n" + "=" * 70)
    print("  PAPER-GRADE METRICS REPORT")
    print("=" * 70)

    # Judge score trend table
    print("\n  1. JUDGE SCORE TREND (1-10) — primary metric over conversation length")
    print(f"  {'System':<16} {'Early (≤T10)':>14} {'Mid (T11-20)':>13} {'Late (>T20)':>12}  {'Slope':>8}")
    print("  " + "-" * 65)
    for s in systems:
        d = data[s]
        t = d["judge_score_trend"]
        slope = d["degradation_slope"]
        e = f"{t['early']:.2f}" if t["early"] is not None else "N/A"
        m = f"{t['mid']:.2f}" if t["mid"] is not None else "N/A"
        l = f"{t['late']:.2f}" if t["late"] is not None else "N/A"
        arrow = "↓" if slope < -0.05 else ("↑" if slope > 0.05 else "→")
        print(f"  {s:<16} {e:>14} {m:>13} {l:>12}  {slope:>+.4f} {arrow}")

    # Degradation by position (normalized, good for plotting)
    cross = data.get("_cross_system", {})
    dbp = cross.get("degradation_by_position", {})
    if dbp:
        print("\n  2. DEGRADATION CURVE BY POSITION (judge score at 20/40/60/80/100% of turns)")
        print(f"  {'System':<16} {'20%':>8} {'40%':>8} {'60%':>8} {'80%':>8} {'100%':>8}")
        print("  " + "-" * 57)
        for s in systems:
            row = dbp.get(s, {})
            vals = [f"{row.get(p, 0) or 0:.2f}" if row.get(p) is not None else " N/A" for p in ["20%","40%","60%","80%","100%"]]
            print(f"  {s:<16} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8}")

    # First violation turn
    print("\n  3. FIRST CONSTRAINT VIOLATION TURN")
    for s in systems:
        fvt = data[s]["first_violation_turn"]
        for cid, turn in fvt.items():
            val = f"Turn {turn}" if turn else "Never violated"
            print(f"  {s:<16} [{cid}]  {val}")

    # CVR per convo
    print("\n  4. CONSTRAINT VIOLATION RATE per conversation")
    for s in systems:
        cvr = data[s]["cvr_per_convo"]
        for cid, rate in cvr.items():
            print(f"  {s:<16} [{cid}]  CVR={rate*100:.1f}%")

    # Per-domain accuracy
    pda = cross.get("per_domain_accuracy", {})
    if pda:
        domains = sorted(set(d for sys_d in pda.values() for d in sys_d))
        if domains:
            print("\n  5. PER-DOMAIN JUDGE SCORE")
            header = f"  {'System':<16}" + "".join(f" {d[:12]:>13}" for d in domains)
            print(header)
            print("  " + "-" * (16 + 13 * len(domains)))
            for s in systems:
                row = pda.get(s, {})
                vals = "".join(f" {row.get(d, 0) or 0:>13.2f}" if row.get(d) is not None else f" {'N/A':>13}" for d in domains)
                print(f"  {s:<16}{vals}")

    # Context utilization + memory efficiency
    print(f"\n  6. CONTEXT BUDGET UTILIZATION (budget={CONTEXT_BUDGET} tokens)")
    print(f"  {'System':<16} {'Avg tokens':>12} {'Avg %':>8} {'Max %':>8}")
    print("  " + "-" * 48)
    for s in systems:
        cu = data[s]["context_utilization"]
        if cu:
            print(f"  {s:<16} {cu['avg_tokens']:>12} {cu['avg_utilization_pct']:>7.1f}% {cu['max_utilization_pct']:>7.1f}%")

    me = cross.get("memory_efficiency")
    if me:
        ratio = me.get("efficiency_ratio")
        saved = me.get("tokens_saved_pct")
        print(f"\n  Memory efficiency (HierMem vs raw_llm):")
        print(f"    HierMem avg tokens:  {me['avg_tokens_hiermem']:,}")
        print(f"    raw_llm avg tokens:  {me['avg_tokens_raw_llm']:,}")
        if ratio is not None:
            direction = "fewer" if ratio < 1.0 else "more"
            print(f"    Efficiency ratio:    {ratio:.3f}  ({abs(saved):.1f}% {direction} tokens)")

    # Response length consistency
    print("\n  7. RESPONSE LENGTH CONSISTENCY")
    print(f"  {'System':<16} {'Avg chars':>12} {'Std dev':>10} {'CV (lower=better)':>20}")
    print("  " + "-" * 62)
    for s in systems:
        rl = data[s]["response_length_stats"]
        if rl:
            print(f"  {s:<16} {rl['avg_chars']:>12} {rl['std_chars']:>10} {rl['cv']:>20.3f}")

    # Latency trend
    print("\n  8. LATENCY TREND (seconds/turn)")
    print(f"  {'System':<16} {'Early (≤T10)':>14} {'Mid (T11-20)':>13} {'Late (>T20)':>12}")
    print("  " + "-" * 58)
    for s in systems:
        lt = data[s]["latency_trend"]
        e = f"{lt['early']:.1f}s" if lt["early"] else "N/A"
        m = f"{lt['mid']:.1f}s" if lt["mid"] else "N/A"
        l = f"{lt['late']:.1f}s" if lt["late"] else "N/A"
        print(f"  {s:<16} {e:>14} {m:>13} {l:>12}")

    # HierMem-specific
    hiermem_data = data.get("hiermem", {})
    if hiermem_data.get("constraint_retrieval_rate") is not None:
        crr = hiermem_data["constraint_retrieval_rate"]
        mt = hiermem_data.get("mode_transition_turn")
        acs = hiermem_data.get("auto_correction_stats") or {}
        print("\n  9. HIERMEM MECHANISM METRICS")
        print(f"  Constraint zone active (% of HYBRID turns): {crr*100:.1f}%")
        print(f"  Mode transition turn (passthrough→HYBRID):  Turn {mt}")
        if acs:
            print(f"  Violation auto-correction trigger rate:     {acs['trigger_rate']*100:.1f}%  "
                  f"({acs['triggered_turns']}/{acs['hybrid_turns']} turns)")
            print(f"  Total violations auto-corrected:            {acs['total_corrections']}")

    # Cost-Quality Ratio
    cqr = cross.get("cost_quality_ratio", {})
    if cqr:
        print("\n  10. COST-QUALITY TRADE-OFF (tokens per quality point)")
        print(f"  {'System':<16} {'Judge Score':>12} {'Avg Tokens':>12} {'Tokens/Point':>14}")
        print("  " + "-" * 58)
        for s in systems:
            row = cqr.get(s, {})
            if row:
                print(f"  {s:<16} {row['avg_quality']:>12.2f} {row['avg_cost_tokens']:>12,} {row['cost_per_quality_point']:>14.0f}")

    # Constraint Retention by Distance
    crbd = cross.get("constraint_retention_by_distance", {})
    if crbd:
        labels = ["T1-10", "T11-20", "T21-30", "T31-40", "T41-50"]
        active_labels = [l for l in labels if any(crbd.get(s, {}).get(l) is not None for s in systems)]
        if active_labels:
            print("\n  11. CONSTRAINT RETENTION BY DISTANCE FROM SETUP (judge score)")
            print(f"      Tests 'Lost in the Middle' — does score degrade with distance?")
            header = f"  {'System':<16}" + "".join(f" {l:>8}" for l in active_labels)
            print(header)
            print("  " + "-" * (16 + 9 * len(active_labels)))
            for s in systems:
                row = crbd.get(s, {})
                vals = "".join(
                    f" {row[l]:>8.2f}" if row.get(l) is not None else f" {'N/A':>8}"
                    for l in active_labels
                )
                print(f"  {s:<16}{vals}")

    # Token-Accuracy summary (just stats, full data in JSON)
    tac = cross.get("token_accuracy_curve", {})
    if tac:
        print("\n  12. TOKEN-ACCURACY SUMMARY (avg tokens at checkpoint turns vs judge score)")
        print(f"  {'System':<16} {'Points':>8} {'Avg Tokens':>12} {'Avg Score':>11}")
        print("  " + "-" * 50)
        for s in systems:
            points = tac.get(s, [])
            if points:
                avg_tok = round(mean(p["tokens"] for p in points))
                avg_sc = round(mean(p["judge_score"] for p in points), 2)
                print(f"  {s:<16} {len(points):>8} {avg_tok:>12,} {avg_sc:>11.2f}")
        print(f"      (Full scatter data available in --json output for plotting)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Extract paper-grade metrics from benchmark results")
    parser.add_argument("results_dir", help="Path to benchmark results directory")
    parser.add_argument("--json", action="store_true", help="Also save metrics to paper_metrics.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_path = results_dir / "results.json"
    metrics_path = results_dir / "metrics_rescored.json"
    if not metrics_path.exists():
        metrics_path = results_dir / "metrics.json"

    if not results_path.exists():
        print(f"ERROR: results.json not found in {results_dir}")
        sys.exit(1)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    metrics_rescored = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    print(f"Loaded: {results_path}")
    print(f"Systems: {list(results.keys())}")
    print(f"Metrics: {metrics_path.name if metrics_path.exists() else 'not found'}")

    data = analyze_all(results, metrics_rescored)
    print_report(data)

    if args.json:
        out_path = results_dir / "paper_metrics.json"
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

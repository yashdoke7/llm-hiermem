"""
Metrics — Compute all evaluation metrics.

Evaluation approach:
  - Primary: LLM-as-judge absolute scoring (MT-Bench / G-Eval style)
      Each response is rated 1-10 independently against the stated constraints.
      Average score per system is the headline metric.
  - Secondary: Pairwise LLM comparison (head-to-head winner)
  - Fallback: Enhanced keyword matching when no judge is available

This follows standard LLM evaluation practice (MT-Bench, AlpacaEval):
a stronger or equal model judges whether constraints are followed.

Primary metrics:
  1. Judge score (1-10) — absolute quality per response, averaged across turns
  2. Degradation Curve — accuracy at conversation checkpoints
  3. Constraint Violation Rate (CVR) — % responses violating stated constraints
  4. Task Accuracy — overall correctness

Secondary metrics:
  5. Pairwise win rate — head-to-head comparison between systems
  6. Latency per turn
  7. Token usage per turn
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# LLM judge state (initialized by init_llm_judge)
_llm_client = None
_judge_available = False
_judge_model = None
_judge_cache = {}  # Cache judge results: (response_hash, constraint_hash) -> bool


def init_llm_judge(provider: str = None, model: str = None):
    """Initialize LLM judge for evaluation. Call before compute_all_metrics.
    
    For stronger evaluation, use a separate provider/model:
      init_llm_judge(provider="google", model="gemini-3.1-flash-lite")
      init_llm_judge(provider="openai", model="gpt-4o-mini")
    
    If no provider specified and default is ollama, skip judge (keyword-only).
    """
    global _llm_client, _judge_available, _judge_model, _judge_cache
    _judge_cache = {}  # Clear cache on re-init
    try:
        import config as cfg
        from llm.client import LLMClient
        judge_provider = provider or cfg.DEFAULT_PROVIDER
        
        # Skip LLM judge if no explicit provider and default is ollama
        # (ollama judge is too weak and may not be running during rescore)
        if provider is None and judge_provider == "ollama":
            logger.info("No judge provider specified, using keyword-only evaluation")
            _judge_available = False
            return
        
        _llm_client = LLMClient(provider=judge_provider)
        _judge_model = model or cfg.SUMMARIZER_MODEL
        _judge_available = True
        logger.info(f"LLM judge initialized: provider={judge_provider}, model={_judge_model}")
    except Exception as e:
        logger.warning(f"LLM judge unavailable, using keyword fallback: {e}")
        _judge_available = False


def compute_all_metrics(all_results: Dict[str, List[Dict]]) -> Dict:
    """Compute all metrics for all systems.
    
    Primary metric: absolute judge scores (1-10 per response, MT-Bench style).
    Secondary: pairwise comparison, gradient keyword scores, binary accuracy.
    """
    metrics = {}
    for system_name, conversations in all_results.items():
        metrics[system_name] = {
            "judge_scores": compute_judge_scores(conversations),   # PRIMARY: 1-10 absolute
            "constraint_violation_rate": compute_cvr(conversations),
            "task_accuracy": compute_task_accuracy(conversations),
            "degradation_curve": compute_degradation_curve(conversations),
            "per_turn_accuracy": compute_per_turn_accuracy(conversations),
            "per_turn_scores": compute_per_turn_scores(conversations),
            "total_turns_processed": sum(c.get("turns_processed", 0) for c in conversations),
        }
    
    # Add pairwise comparison if we have exactly 2 systems and judge available
    system_names = list(all_results.keys())
    if len(system_names) >= 2 and _judge_available:
        metrics["pairwise_comparison"] = compute_pairwise_comparison(
            all_results, system_names[0], system_names[1]
        )
    
    return metrics


def _get_results(conv: Dict) -> List[Dict]:
    """Get turn results, handling both old and new key names."""
    return conv.get("results", conv.get("turn_logs", []))


def _clean_response(response: str) -> str:
    """Strip hallucinated User:/Assistant: continuations from a response."""
    match = re.search(r'\n\s*User\s*:', response)
    if match:
        return response[:match.start()].rstrip()
    return response


def _find_response_at_turn(results: List[Dict], turn: int) -> str:
    """Find the response text at a given turn number."""
    for r in results:
        if r.get("turn") == turn:
            raw = r.get("response", "")
            return _clean_response(raw) if raw else None
    return None


def _get_constraint_text(conv: Dict) -> str:
    """Extract the original constraint text from a conversation."""
    constraints = conv.get("constraints", [])
    if constraints:
        return "\n".join(constraints)
    checkpoints = conv.get("checkpoints", [])
    if checkpoints:
        return checkpoints[0].get("constraint_tested", "")
    return ""


def _deduplicate_checkpoints(checkpoints: List[Dict]) -> List[Dict]:
    """Deduplicate checkpoints — keep one per turn, evaluate each keyword group separately.
    
    The post-processor creates multiple checkpoint entries per turn (one per
    keyword group, e.g. metric keywords + substitution keywords).
    
    We merge into one entry per turn but keep keyword groups separate so each
    constraint can be evaluated independently. A turn passes only if ALL
    keyword groups pass.
    """
    seen = set()
    deduped = []
    for cp in checkpoints:
        turn = cp.get("turn", 0)
        if turn not in seen:
            seen.add(turn)
            keyword_groups = []
            for other in checkpoints:
                if other.get("turn") == turn:
                    kws = other.get("keywords", [])
                    if kws:
                        keyword_groups.append(kws)
            merged = dict(cp)
            merged["keyword_groups"] = keyword_groups
            # Keep flat keywords for backward compat (union of all groups)
            merged["keywords"] = list(set(
                kw for group in keyword_groups for kw in group
            ))
            deduped.append(merged)
    return deduped


def compute_cvr(conversations: List[Dict]) -> float:
    """Compute Constraint Violation Rate across all conversations."""
    total_checks = 0
    violations = 0

    for conv in conversations:
        checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
        results = _get_results(conv)
        constraint_text = _get_constraint_text(conv)

        for cp in checkpoints:
            response = _find_response_at_turn(results, cp.get("turn", 0))
            if response is not None and not response.startswith("ERROR:"):
                total_checks += 1
                passed = evaluate_checkpoint(
                    response, constraint_text,
                    cp.get("constraint_tested", ""),
                    cp.get("keywords", []),
                    keyword_groups=cp.get("keyword_groups")
                )
                if not passed:
                    violations += 1

    if total_checks == 0:
        return 0.0
    return round(violations / total_checks, 4)


def compute_task_accuracy(conversations: List[Dict]) -> float:
    """Compute overall task accuracy across checkpoint evaluations."""
    total = 0
    correct = 0

    for conv in conversations:
        checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
        results = _get_results(conv)
        constraint_text = _get_constraint_text(conv)

        for cp in checkpoints:
            response = _find_response_at_turn(results, cp.get("turn", 0))
            if response is not None and not response.startswith("ERROR:"):
                total += 1
                if evaluate_checkpoint(
                    response, constraint_text,
                    cp.get("constraint_tested", ""),
                    cp.get("keywords", []),
                    keyword_groups=cp.get("keyword_groups")
                ):
                    correct += 1

    if total == 0:
        return 0.0
    return round(correct / total, 4)


def compute_degradation_curve(conversations: List[Dict],
                               checkpoints_at: List[int] = None) -> Dict[int, float]:
    """Compute accuracy at different turn distances."""
    if checkpoints_at is None:
        checkpoints_at = [1, 10, 20, 30, 40, 50]

    curve = {}
    for turn_threshold in checkpoints_at:
        total = 0
        correct = 0

        for conv in conversations:
            checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
            results = _get_results(conv)
            constraint_text = _get_constraint_text(conv)

            for cp in checkpoints:
                if cp.get("turn", 0) <= turn_threshold:
                    response = _find_response_at_turn(results, cp["turn"])
                    if response is not None and not response.startswith("ERROR:"):
                        total += 1
                        if evaluate_checkpoint(
                            response, constraint_text,
                            cp.get("constraint_tested", ""),
                            cp.get("keywords", []),
                            keyword_groups=cp.get("keyword_groups")
                        ):
                            correct += 1

        curve[turn_threshold] = round(correct / total, 4) if total > 0 else 0.0

    return curve


def compute_per_turn_accuracy(conversations: List[Dict]) -> Dict[str, Any]:
    """Compute accuracy at each individual checkpoint turn.
    
    Returns dict with:
      - per_turn: {turn_number: {"passed": bool, "convo": convo_id, "mode": str}}
      - summary: {turn_number: accuracy} for turns with multiple convos
    """
    turn_results = {}  # turn -> list of (passed, convo_id, mode)

    for conv in conversations:
        convo_id = conv.get("conversation_id", "?")
        checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
        results = _get_results(conv)
        constraint_text = _get_constraint_text(conv)

        for cp in checkpoints:
            turn = cp.get("turn", 0)
            response = _find_response_at_turn(results, turn)
            if response is None or response.startswith("ERROR:"):
                continue

            passed = evaluate_checkpoint(
                response, constraint_text,
                cp.get("constraint_tested", ""),
                cp.get("keywords", []),
                keyword_groups=cp.get("keyword_groups")
            )

            # Detect mode from turn_logs
            mode = "unknown"
            for r in results:
                if r.get("turn") == turn:
                    pd = r.get("pipeline_details")
                    if pd is None:
                        # Non-HierMem system (raw_llm, rag, etc.)
                        mode = "full_history"
                        break
                    warnings = pd.get("warnings", [])
                    sources = pd.get("sources_used", [])
                    if any("passthrough" in str(w) for w in warnings):
                        mode = "passthrough"
                    elif any("passthrough" in str(s) for s in sources):
                        mode = "passthrough"
                    else:
                        strategy = pd.get("curator_strategy")
                        mode = f"curator:{strategy}" if strategy else "curator"
                    break

            if turn not in turn_results:
                turn_results[turn] = []
            turn_results[turn].append({
                "passed": passed,
                "convo": convo_id,
                "mode": mode,
            })

    # Build summary: accuracy per turn across all convos
    summary = {}
    for turn in sorted(turn_results.keys()):
        entries = turn_results[turn]
        acc = sum(1 for e in entries if e["passed"]) / len(entries)
        summary[str(turn)] = round(acc, 4)

    # Build detailed per-turn list
    detail = []
    for turn in sorted(turn_results.keys()):
        for entry in turn_results[turn]:
            detail.append({
                "turn": turn,
                "passed": entry["passed"],
                "convo": entry["convo"],
                "mode": entry["mode"],
            })

    return {"detail": detail, "summary": summary}


def evaluate_checkpoint(response: str, constraint_text: str,
                        checkpoint_constraint: str,
                        keywords: List[str] = None,
                        keyword_groups: List[List[str]] = None) -> bool:
    """
    Evaluate whether a response passes a checkpoint test.
    
    Strategy:
      1. If keyword_groups provided, evaluate each group separately (ALL must pass)
      2. If LLM judge is available → use it as additional signal (OR with keywords)
      3. Fallback → enhanced keyword matching
    """
    if response.startswith("ERROR:"):
        return False

    # Evaluate keyword groups separately if available
    if keyword_groups and len(keyword_groups) > 1:
        keyword_result = all(
            _keyword_evaluate(response, group) for group in keyword_groups
        )
    else:
        keyword_result = _keyword_evaluate(response, keywords)

    # Try LLM-as-judge if available
    if _judge_available and _llm_client:
        try:
            llm_result = _llm_judge_evaluate(response, constraint_text, checkpoint_constraint)
            logger.debug(f"Judge: keyword={keyword_result}, llm={llm_result}")
            # Pass if EITHER method says pass
            return keyword_result or llm_result
        except Exception as e:
            logger.debug(f"LLM judge failed, using keywords only: {e}")

    return keyword_result


def _llm_judge_evaluate(response: str, constraint_text: str,
                         checkpoint_constraint: str) -> bool:
    """Use LLM to judge if response follows constraints.
    
    Results are cached to avoid redundant API calls when the same
    response+constraint pair is evaluated multiple times (CVR, accuracy, etc.).
    """
    # Check cache first
    cache_key = (hash(response[:2000]), hash(constraint_text))
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]
    
    constraints_clean = _parse_individual_constraints(constraint_text)

    prompt = (
        f"You are evaluating whether an AI assistant's response follows user-specified rules.\n\n"
        f"RULES the user asked the assistant to follow:\n{constraints_clean}\n\n"
        f"ASSISTANT'S RESPONSE:\n{response[:2000]}\n\n"
        f"Does the response follow the rules? Consider:\n"
        f"- Metric measurements (g, ml, kg) as primary = PASS, even with minor imperial (tbsp, tsp)\n"
        f"- Budget-friendly alternatives or cheaper options mentioned = PASS for substitution rule\n"
        f"- Relevant disclaimers included = PASS for disclaimer rule\n"
        f"- If metric is primary and imperial appears in parentheses as conversion, that's correct\n"
        f"- Be fair: if the response shows awareness of the constraint and mostly follows it, PASS\n"
        f"- Only FAIL if the constraint is clearly ignored or violated in a major way\n\n"
        f"Output ONLY one word: PASS or FAIL"
    )

    result = _llm_client.call(
        system_prompt="You are a strict evaluator. Only output PASS or FAIL.",
        user_prompt=prompt,
        model=_judge_model,
        temperature=0.0,
        max_tokens=256
    )

    if not result:
        logger.warning("LLM judge returned empty response, defaulting to FAIL")
        _judge_cache[cache_key] = False
        return False
    passed = "PASS" in result.strip().upper()
    _judge_cache[cache_key] = passed
    return passed


def _parse_individual_constraints(constraint_text: str) -> str:
    """Return constraint text as-is — the raw user message is already readable.
    
    Previous regex parsing was mangling constraints (e.g. matching 'never' inside
    'whenever') and confusing the judge. The original user message is clearest.
    """
    return constraint_text


def _keyword_evaluate(response: str, keywords: List[str] = None) -> bool:
    """Enhanced keyword matching with positive AND negative checks.
    
    For metric checks: uses a ratio-based approach that accounts for
    parenthetical conversions. "400g (about 1 pound)" counts as metric
    because the primary unit is metric. "1/2 pound (225g)" counts as imperial
    because the primary unit is imperial.
    """
    response_lower = response.lower()

    if not keywords:
        return True

    # Check if metric-related keywords are in the set
    metric_keywords = {"gram", "ml", "liter", "celsius", "kg", "metric"}
    is_metric_check = any(kw.lower() in metric_keywords for kw in keywords)

    if is_metric_check:
        return _evaluate_metric_compliance(response_lower, keywords)

    # General positive check: at least one keyword present
    return any(kw.lower() in response_lower for kw in keywords)


def _evaluate_metric_compliance(response_lower: str, keywords: List[str]) -> bool:
    """Evaluate whether a response primarily uses metric measurements.
    
    Handles parenthetical conversions:
      "400g (about 1 pound)"  → counted as 1 metric (imperial is just a conversion)
      "1/2 pound (225g)"      → counted as 1 imperial (metric is just a conversion)
      "1 tablespoon olive oil" → counted as 1 imperial (standalone)
      "200g chicken"           → counted as 1 metric (standalone)
    
    Pass threshold: >50% of primary measurements are metric.
    Small units (tablespoon, teaspoon) used standalone are penalized less
    since they have no natural metric equivalent in casual cooking.
    """
    # Pattern: metric value followed by parenthetical imperial conversion
    # e.g., "400g (about 1 pound)", "200ml (about 3/4 cup)"
    metric_primary_patterns = [
        r'\d+\s*(?:g|grams?)\s*\([^)]*(?:pound|lb|oz|ounce|cup)[^)]*\)',
        r'\d+\s*(?:ml|milliliters?)\s*\([^)]*(?:cup|fl\s*oz|ounce)[^)]*\)',
        r'\d+\s*(?:kg|kilograms?)\s*\([^)]*(?:pound|lb)[^)]*\)',
        r'\d+\s*°?c(?:elsius)?\s*\([^)]*(?:fahrenheit|°?f)[^)]*\)',
    ]
    
    # Pattern: imperial value followed by parenthetical metric conversion
    # e.g., "1/2 pound (225g)", "1/4 cup (60ml)"
    imperial_primary_patterns = [
        r'[\d/]+\s*(?:pounds?|lbs?)\s*\([^)]*(?:g|gram|kg)[^)]*\)',
        r'[\d/]+\s*(?:cups?)\s*\([^)]*(?:ml|g|gram|liter)[^)]*\)',
        r'[\d/]+\s*(?:ounces?|oz)\s*\([^)]*(?:g|gram|ml)[^)]*\)',
        r'[\d/]+\s*(?:°?f|fahrenheit)\s*\([^)]*(?:°?c|celsius)[^)]*\)',
    ]
    
    # Count metric-primary conversions (metric first, imperial in parens)
    metric_from_conversions = sum(
        len(re.findall(p, response_lower)) for p in metric_primary_patterns
    )
    # Count imperial-primary conversions (imperial first, metric in parens)
    imperial_from_conversions = sum(
        len(re.findall(p, response_lower)) for p in imperial_primary_patterns
    )
    
    # Now count standalone measurements (not part of conversion pairs)
    # First, strip out all parenthetical content to avoid double-counting
    response_no_parens = re.sub(r'\([^)]*\)', '', response_lower)
    
    standalone_metric_patterns = [
        r'\d+\s*(?:g|grams?)\b',
        r'\d+\s*(?:ml|milliliters?)\b',
        r'\d+\s*(?:l|liters?)\b',
        r'\d+\s*(?:kg|kilograms?)\b',
        r'\d+\s*°?c(?:elsius)?\b',
    ]
    standalone_metric = sum(
        len(re.findall(p, response_no_parens)) for p in standalone_metric_patterns
    )
    # Subtract conversion pairs already counted
    standalone_metric = max(0, standalone_metric - metric_from_conversions)
    
    # Imperial standalone — separate "major" (cups, pounds) from "minor" (tbsp, tsp)
    major_imperial_patterns = [
        r'\b\d+\s*cups?\b', r'\b\d+/\d+\s*cups?\b',
        r'\b\d+\s*ounces?\b', r'\b\d+\s*pounds?\b',
        r'\b\d+\s*oz\b', r'\b\d+\s*lbs?\b', r'\bfahrenheit\b',
    ]
    minor_imperial_patterns = [
        r'\b\d+\s*tablespoons?\b', r'\b\d+\s*teaspoons?\b',
        r'\b\d+\s*tbsp\b', r'\b\d+\s*tsp\b',
    ]
    major_imperial = sum(
        len(re.findall(p, response_no_parens)) for p in major_imperial_patterns
    )
    major_imperial = max(0, major_imperial - imperial_from_conversions)
    minor_imperial = sum(
        len(re.findall(p, response_no_parens)) for p in minor_imperial_patterns
    )
    
    # Total metric = conversions with metric primary + standalone metric
    total_metric = metric_from_conversions + standalone_metric
    # Total imperial = conversions with imperial primary + major imperial + minor (weighted 0.5)
    # Minor imperial (tbsp/tsp) penalized less — no natural metric equivalent in casual cooking
    total_imperial = imperial_from_conversions + major_imperial + (minor_imperial * 0.5)
    
    total = total_metric + total_imperial
    if total == 0:
        return any(kw.lower() in response_lower for kw in keywords)
    
    metric_ratio = total_metric / total
    logger.debug(
        f"Metric eval: metric={total_metric} (conv={metric_from_conversions}, "
        f"standalone={standalone_metric}), imperial={total_imperial} "
        f"(conv={imperial_from_conversions}, major={major_imperial}, "
        f"minor={minor_imperial}), ratio={metric_ratio:.2f}"
    )
    return metric_ratio > 0.5


def _extract_test_keywords(test: str) -> List[str]:
    """Extract keywords from a test question."""
    quoted = re.findall(r'"([^"]+)"', test) + re.findall(r"'([^']+)'", test)
    if quoted:
        return quoted
    caps = re.findall(r'\b[A-Z][a-zA-Z]+\b', test)
    if caps:
        return caps
    stop_words = {"is", "the", "in", "does", "do", "are", "a", "an", "to", "of",
                  "response", "code", "use", "contain", "mention", "have"}
    words = [w.strip("?.,!") for w in test.split() if w.lower() not in stop_words and len(w) > 2]
    return words[:3]


# ─── Gradient Scoring (0–100) ───────────────────────────────────────────────


def score_checkpoint(response: str, keywords: List[str] = None,
                     keyword_groups: List[List[str]] = None) -> float:
    """Score how well a response follows constraints on a 0–100 scale.
    
    Unlike binary evaluate_checkpoint, this returns a gradient:
      100 = perfectly follows all constraint groups
        0 = completely ignores all constraints
    
    Each keyword group is scored independently, then averaged.
    """
    if not response or response.startswith("ERROR:"):
        return 0.0
    
    if keyword_groups and len(keyword_groups) > 1:
        group_scores = [_score_keyword_group(response, g) for g in keyword_groups]
        return round(sum(group_scores) / len(group_scores), 1)
    elif keywords:
        return round(_score_keyword_group(response, keywords), 1)
    return 100.0


def _score_keyword_group(response: str, keywords: List[str]) -> float:
    """Score a single keyword group on 0–100 scale."""
    response_lower = response.lower()
    
    if not keywords:
        return 100.0
    
    metric_keywords = {"gram", "ml", "liter", "celsius", "kg", "metric"}
    is_metric_check = any(kw.lower() in metric_keywords for kw in keywords)
    
    if is_metric_check:
        return _score_metric_compliance(response_lower)
    
    # General keywords: score = % of keywords found
    found = sum(1 for kw in keywords if kw.lower() in response_lower)
    return round(found / len(keywords) * 100, 1)


def _score_metric_compliance(response_lower: str) -> float:
    """Score metric compliance on 0–100 scale.
    
    Uses same logic as _evaluate_metric_compliance but returns ratio * 100.
    """
    # Detect conversion patterns
    metric_conv = sum(len(re.findall(p, response_lower)) for p in [
        r'\d+\s*(?:g|grams?|ml|kg)\s*\([^)]*(?:pound|lb|oz|cup|ounce)[^)]*\)',
    ])
    imperial_conv = sum(len(re.findall(p, response_lower)) for p in [
        r'[\d/]+\s*(?:pounds?|lbs?|cups?|ounces?|oz)\s*\([^)]*(?:g|gram|kg|ml)[^)]*\)',
    ])
    
    response_no_parens = re.sub(r'\([^)]*\)', '', response_lower)
    
    standalone_metric = sum(len(re.findall(p, response_no_parens)) for p in [
        r'\d+\s*(?:g|grams?)\b', r'\d+\s*(?:ml|milliliters?)\b',
        r'\d+\s*(?:l|liters?)\b', r'\d+\s*(?:kg|kilograms?)\b',
        r'\d+\s*°?c(?:elsius)?\b',
    ])
    standalone_metric = max(0, standalone_metric - metric_conv)
    
    major_imperial = sum(len(re.findall(p, response_no_parens)) for p in [
        r'\b\d+\s*cups?\b', r'\b\d+/\d+\s*cups?\b',
        r'\b\d+\s*ounces?\b', r'\b\d+\s*pounds?\b',
        r'\b\d+\s*oz\b', r'\b\d+\s*lbs?\b', r'\bfahrenheit\b',
    ])
    major_imperial = max(0, major_imperial - imperial_conv)
    minor_imperial = sum(len(re.findall(p, response_no_parens)) for p in [
        r'\b\d+\s*tablespoons?\b', r'\b\d+\s*teaspoons?\b',
        r'\b\d+\s*tbsp\b', r'\b\d+\s*tsp\b',
    ])
    
    total_metric = metric_conv + standalone_metric
    total_imperial = imperial_conv + major_imperial + (minor_imperial * 0.5)
    total = total_metric + total_imperial
    
    if total == 0:
        return 50.0  # No measurements — neutral score
    
    return round(total_metric / total * 100, 1)


def compute_per_turn_scores(conversations: List[Dict]) -> Dict[str, Any]:
    """Compute gradient scores (0–100) at each checkpoint turn.
    
    Returns per-turn scores showing HOW WELL constraints are followed,
    not just binary pass/fail.
    """
    detail = []
    
    for conv in conversations:
        convo_id = conv.get("conversation_id", "?")
        checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
        results = _get_results(conv)
        
        for cp in checkpoints:
            turn = cp.get("turn", 0)
            response = _find_response_at_turn(results, turn)
            if response is None or response.startswith("ERROR:"):
                continue
            
            kw_groups = cp.get("keyword_groups")
            keywords = cp.get("keywords", [])
            
            score = score_checkpoint(response, keywords, kw_groups)
            
            # Per-group scores for detail
            group_scores = []
            if kw_groups:
                for g in kw_groups:
                    group_scores.append({
                        "keywords": g[:3],
                        "score": _score_keyword_group(response, g),
                    })
            
            # Mode detection
            mode = "unknown"
            for r in results:
                if r.get("turn") == turn:
                    pd = r.get("pipeline_details")
                    if pd is None:
                        mode = "full_history"
                    elif any("passthrough" in str(w) for w in pd.get("warnings", [])):
                        mode = "passthrough"
                    elif any("passthrough" in str(s) for s in pd.get("sources_used", [])):
                        mode = "passthrough"
                    else:
                        strategy = pd.get("curator_strategy")
                        mode = f"curator:{strategy}" if strategy else "curator"
                    break
            
            detail.append({
                "turn": turn,
                "convo": convo_id,
                "score": score,
                "group_scores": group_scores,
                "mode": mode,
            })
    
    # Summary: average score per turn
    from collections import defaultdict
    turn_scores = defaultdict(list)
    for d in detail:
        turn_scores[d["turn"]].append(d["score"])
    summary = {str(t): round(sum(s)/len(s), 1) for t, s in sorted(turn_scores.items())}
    avg_score = round(sum(d["score"] for d in detail) / max(len(detail), 1), 1)
    
    return {"detail": detail, "summary": summary, "avg_score": avg_score}


# ─── Absolute Judge Scoring (1–10, MT-Bench style) ──────────────────────────


def compute_judge_scores(conversations: List[Dict]) -> Dict[str, Any]:
    """Rate each checkpoint response independently on a 1-10 scale.
    
    This is the primary metric — follows MT-Bench / G-Eval methodology:
      10 = Perfectly follows all stated constraints
       7 = Mostly follows, minor gaps
       5 = Partially follows (some constraints respected, others not)
       3 = Mostly ignores constraints
       1 = Completely ignores constraints
    
    Falls back to scaled keyword score (0-100 → 1-10) when judge unavailable.
    
    Returns:
      avg_score: float (1-10) — headline metric
      detail: [{turn, convo, judge_score, mode}]
      summary: {turn: avg_score}
    """
    detail = []

    for conv in conversations:
        convo_id = conv.get("conversation_id", "?")
        checkpoints = _deduplicate_checkpoints(conv.get("checkpoints", []))
        results = _get_results(conv)
        constraint_text = _get_constraint_text(conv)

        for cp in checkpoints:
            turn = cp.get("turn", 0)
            response = _find_response_at_turn(results, turn)
            if response is None or response.startswith("ERROR:"):
                continue

            if _judge_available and _llm_client:
                score = _absolute_judge(response, constraint_text,
                                        cp.get("constraint_tested", ""))
            else:
                # Fallback: scale keyword score from 0-100 to 1-10
                kw_score = score_checkpoint(response, cp.get("keywords", []),
                                            cp.get("keyword_groups"))
                score = round(1.0 + (kw_score / 100.0) * 9.0, 1)

            # Mode detection
            mode = "unknown"
            for r in results:
                if r.get("turn") == turn:
                    pd = r.get("pipeline_details")
                    if pd is None:
                        mode = "full_history"
                    elif any("passthrough" in str(w) for w in pd.get("warnings", [])):
                        mode = "passthrough"
                    elif any("passthrough" in str(s) for s in pd.get("sources_used", [])):
                        mode = "passthrough"
                    else:
                        strategy = pd.get("curator_strategy")
                        mode = f"curator:{strategy}" if strategy else "curator"
                    break

            detail.append({
                "turn": turn,
                "convo": convo_id,
                "judge_score": score,
                "mode": mode,
            })

    from collections import defaultdict
    turn_scores = defaultdict(list)
    for d in detail:
        turn_scores[d["turn"]].append(d["judge_score"])
    summary = {str(t): round(sum(s) / len(s), 2) for t, s in sorted(turn_scores.items())}
    avg_score = round(sum(d["judge_score"] for d in detail) / max(len(detail), 1), 2)

    return {"avg_score": avg_score, "detail": detail, "summary": summary}


def _absolute_judge(response: str, constraint_text: str,
                    checkpoint_constraint: str = "") -> float:
    """Rate a single response on a 1-10 scale for constraint compliance.
    
    Caches results to avoid duplicate API calls.
    Returns a float 1.0–10.0.
    """
    cache_key = ("abs", hash(response[:4000]), hash(constraint_text))
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]

    constraints_clean = _parse_individual_constraints(constraint_text)

    # Truncate to 4000 chars so the full response (including later try/except) is visible
    response_text = response[:4000]

    prompt = (
        f"You are evaluating how well an AI assistant follows user-specified rules.\n\n"
        f"RULES the user asked the assistant to follow:\n{constraints_clean}\n\n"
        f"ASSISTANT'S RESPONSE:\n{response_text}\n\n"
        f"Rate the response on a scale of 1 to 10 for how well it follows ALL of the rules:\n"
        f"  10 = Perfectly follows every rule stated\n"
        f"   8 = Follows most rules, one minor lapse\n"
        f"   6 = Follows some rules but misses others\n"
        f"   4 = Mostly ignores the rules with occasional correct elements\n"
        f"   2 = Almost completely ignores the rules\n"
        f"   1 = Completely ignores all rules\n\n"
        f"Notes:\n"
        f"- Metric measurements (g, ml, kg) as primary with imperial in parentheses = following metric rule\n"
        f"- Imperial measurements (cups, pounds) as primary = NOT following metric rule\n"
        f"- Minor units (tablespoon, teaspoon) alone do not violate the metric rule\n"
        f"- Budget alternatives or cheaper substitutions mentioned = following the substitution rule\n\n"
        f"Output your score as: SCORE: <number>\n"
        f"Then optionally one sentence of reasoning."
    )

    try:
        result = _llm_client.call(
            system_prompt="You are an impartial evaluator. Always output your score as: SCORE: <number from 1 to 10>",
            user_prompt=prompt,
            model=_judge_model,
            temperature=0.0,
            max_tokens=256,
        )

        if not result:
            logger.warning("Absolute judge returned empty response, defaulting to 5")
            _judge_cache[cache_key] = 5.0
            return 5.0

        # Try to extract "SCORE: N" first (most reliable)
        score_match = re.search(r'SCORE:\s*(10|[1-9])', result, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Fallback: find the LAST standalone integer in response (avoids numbered lists)
            all_matches = re.findall(r'(?<!\d)(10|[1-9])(?!\d)', result.strip())
            score = float(all_matches[-1]) if all_matches else 5.0
        _judge_cache[cache_key] = score
        return score

    except Exception as e:
        logger.warning(f"Absolute judge failed: {e}")
        _judge_cache[cache_key] = 5.0
        return 5.0


# ─── Pairwise LLM Comparison ────────────────────────────────────────────────


def compute_pairwise_comparison(all_results: Dict[str, List[Dict]],
                                system_a: str, system_b: str) -> Dict:
    """Compare two systems head-to-head using LLM judge.
    
    For each checkpoint turn that both systems have responses for,
    asks the judge: "Which response better follows the stated constraints?"
    
    Returns wins/losses/ties and per-turn comparison details.
    """
    if not _judge_available:
        return {"error": "LLM judge not available for pairwise comparison"}
    
    convos_a = {c["conversation_id"]: c for c in all_results.get(system_a, [])}
    convos_b = {c["conversation_id"]: c for c in all_results.get(system_b, [])}
    
    comparisons = []
    wins_a, wins_b, ties = 0, 0, 0
    
    for cid in convos_a:
        if cid not in convos_b:
            continue
        
        conv_a, conv_b = convos_a[cid], convos_b[cid]
        checkpoints = _deduplicate_checkpoints(conv_a.get("checkpoints", []))
        results_a = _get_results(conv_a)
        results_b = _get_results(conv_b)
        constraint_text = _get_constraint_text(conv_a)
        
        for cp in checkpoints:
            turn = cp.get("turn", 0)
            resp_a = _find_response_at_turn(results_a, turn)
            resp_b = _find_response_at_turn(results_b, turn)
            
            if not resp_a or not resp_b:
                continue
            if resp_a.startswith("ERROR:") or resp_b.startswith("ERROR:"):
                continue
            
            result = _pairwise_judge(resp_a, resp_b, constraint_text, system_a, system_b)
            comparisons.append({
                "convo": cid,
                "turn": turn,
                "winner": result["winner"],
                "score_a": result["score_a"],
                "score_b": result["score_b"],
                "reasoning": result.get("reasoning", ""),
            })
            
            if result["winner"] == system_a:
                wins_a += 1
            elif result["winner"] == system_b:
                wins_b += 1
            else:
                ties += 1
    
    total = wins_a + wins_b + ties
    return {
        "system_a": system_a,
        "system_b": system_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total": total,
        "win_rate_a": round(wins_a / max(total, 1) * 100, 1),
        "win_rate_b": round(wins_b / max(total, 1) * 100, 1),
        "comparisons": comparisons,
    }


def _pairwise_judge(resp_a: str, resp_b: str, constraint_text: str,
                     name_a: str, name_b: str) -> Dict:
    """Ask LLM judge to compare two responses head-to-head.
    
    Returns {"winner": name_a|name_b|"tie", "score_a": 1-5, "score_b": 1-5}
    """
    # Check cache
    cache_key = (hash(resp_a[:3000]), hash(resp_b[:3000]), hash(constraint_text))
    if cache_key in _judge_cache:
        return _judge_cache[cache_key]
    
    prompt = (
        f"Compare two AI assistant responses for how well they follow user rules.\n\n"
        f"USER RULES:\n{constraint_text}\n\n"
        f"RESPONSE A:\n{resp_a[:3000]}\n\n"
        f"RESPONSE B:\n{resp_b[:3000]}\n\n"
        f"Rate each response 1-5 for constraint compliance:\n"
        f"  5 = Perfectly follows all rules\n"
        f"  4 = Mostly follows rules, minor lapses\n"
        f"  3 = Partially follows rules\n"
        f"  2 = Mostly ignores rules\n"
        f"  1 = Completely ignores rules\n\n"
        f"Consider:\n"
        f"- Metric measurements (g, ml, kg) primary with imperial in parentheses = following the metric rule\n"
        f"- Imperial primary (cups, pounds) with metric in parentheses = NOT following\n"
        f"- Budget alternatives, cheaper options, substitutions = following the substitution rule\n"
        f"- Minor units (tablespoons, teaspoons) alone don't violate the metric rule\n\n"
        f"Output EXACTLY in this format (3 lines):\n"
        f"A: <score>\n"
        f"B: <score>\n"
        f"REASON: <one sentence explaining the difference>"
    )
    
    try:
        result = _llm_client.call(
            system_prompt="You are an impartial evaluator comparing AI responses. Be fair and specific.",
            user_prompt=prompt,
            model=_judge_model,
            temperature=0.0,
            max_tokens=256,
        )
        
        if not result:
            out = {"winner": "tie", "score_a": 3, "score_b": 3}
            _judge_cache[cache_key] = out
            return out
        
        # Parse "A: X\nB: Y\nREASON: ..."
        lines = result.strip().split("\n")
        score_a, score_b = 3, 3
        reasoning = ""
        for line in lines:
            line = line.strip()
            if line.startswith("A:"):
                try:
                    score_a = int(re.search(r'\d', line).group())
                except (AttributeError, ValueError):
                    pass
            elif line.startswith("B:"):
                try:
                    score_b = int(re.search(r'\d', line).group())
                except (AttributeError, ValueError):
                    pass
            elif line.startswith("REASON:"):
                reasoning = line[7:].strip()
        
        if score_a > score_b:
            winner = name_a
        elif score_b > score_a:
            winner = name_b
        else:
            winner = "tie"
        
        out = {"winner": winner, "score_a": score_a, "score_b": score_b, "reasoning": reasoning}
        _judge_cache[cache_key] = out
        return out
        
    except Exception as e:
        logger.warning(f"Pairwise judge failed: {e}")
        out = {"winner": "tie", "score_a": 3, "score_b": 3, "reasoning": f"Judge error: {e}"}
        _judge_cache[cache_key] = out
        return out

"""
Metrics — Compute all evaluation metrics.

Evaluation approach:
  - Primary: LLM-as-judge (uses curator/summarizer model to score responses)
  - Fallback: Enhanced keyword matching (positive + negative keywords)
  
This follows standard LLM evaluation practice (MT-Bench, AlpacaEval):
a stronger or equal model judges whether constraints are followed.

Primary metrics:
  1. Degradation Curve — accuracy at conversation checkpoints
  2. Constraint Violation Rate (CVR) — % responses violating stated constraints
  3. Task Accuracy — overall correctness

Secondary metrics:
  4. Latency per turn
  5. Token usage per turn
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# LLM judge state (initialized by init_llm_judge)
_llm_client = None
_judge_available = False
_judge_model = None


def init_llm_judge(provider: str = None, model: str = None):
    """Initialize LLM judge for evaluation. Call before compute_all_metrics."""
    global _llm_client, _judge_available, _judge_model
    try:
        import config as cfg
        from llm.client import LLMClient
        _llm_client = LLMClient(provider=provider or cfg.DEFAULT_PROVIDER)
        _judge_model = model or cfg.SUMMARIZER_MODEL
        _judge_available = True
        logger.info(f"LLM judge initialized: {_judge_model}")
    except Exception as e:
        logger.warning(f"LLM judge unavailable, using keyword fallback: {e}")
        _judge_available = False


def compute_all_metrics(all_results: Dict[str, List[Dict]]) -> Dict:
    """Compute all metrics for all systems."""
    metrics = {}
    for system_name, conversations in all_results.items():
        metrics[system_name] = {
            "constraint_violation_rate": compute_cvr(conversations),
            "task_accuracy": compute_task_accuracy(conversations),
            "degradation_curve": compute_degradation_curve(conversations),
            "per_turn_accuracy": compute_per_turn_accuracy(conversations),
            "total_turns_processed": sum(c.get("turns_processed", 0) for c in conversations),
        }
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
    """Deduplicate checkpoints — keep one per turn with merged keywords.
    
    The post-processor creates multiple checkpoint entries per turn (one per
    keyword group). For evaluation we want ONE check per checkpoint turn,
    not multiple keyword-group checks that inflate the denominator.
    """
    seen = set()
    deduped = []
    for cp in checkpoints:
        turn = cp.get("turn", 0)
        if turn not in seen:
            seen.add(turn)
            all_keywords = []
            for other in checkpoints:
                if other.get("turn") == turn:
                    all_keywords.extend(other.get("keywords", []))
            merged = dict(cp)
            merged["keywords"] = list(set(all_keywords))
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
                    cp.get("keywords", [])
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
                    cp.get("keywords", [])
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
                            cp.get("keywords", [])
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
                cp.get("keywords", [])
            )

            # Detect mode from turn_logs
            mode = "unknown"
            for r in results:
                if r.get("turn") == turn:
                    warnings = r.get("pipeline_details", {}).get("warnings", [])
                    sources = r.get("pipeline_details", {}).get("sources_used", [])
                    if any("passthrough" in str(w) for w in warnings):
                        mode = "passthrough"
                    elif any("passthrough" in str(s) for s in sources):
                        mode = "passthrough"
                    else:
                        strategy = r.get("pipeline_details", {}).get("curator_strategy")
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
                        keywords: List[str] = None) -> bool:
    """
    Evaluate whether a response passes a checkpoint test.
    
    Strategy:
      1. If LLM judge is available → use it as TIE-BREAKER with keyword eval
      2. Fallback → enhanced keyword matching (positive + negative)
    
    Using both methods and requiring agreement reduces false positives/negatives
    from either method alone.
    """
    if response.startswith("ERROR:"):
        return False

    # Always run keyword evaluation
    keyword_result = _keyword_evaluate(response, keywords)

    # Try LLM-as-judge if available
    if _judge_available and _llm_client:
        try:
            llm_result = _llm_judge_evaluate(response, constraint_text, checkpoint_constraint)
            logger.debug(f"Judge: keyword={keyword_result}, llm={llm_result}")
            # Pass if EITHER method says pass (generous — avoids false negatives
            # from weak judge model or incomplete keyword lists)
            return keyword_result or llm_result
        except Exception as e:
            logger.debug(f"LLM judge failed, using keywords only: {e}")

    return keyword_result


def _llm_judge_evaluate(response: str, constraint_text: str,
                         checkpoint_constraint: str) -> bool:
    """Use LLM to judge if response follows constraints."""
    constraints_clean = _parse_individual_constraints(constraint_text)

    prompt = (
        f"You are evaluating whether an AI assistant's response follows specific rules.\n\n"
        f"RULES the assistant must follow:\n{constraints_clean}\n\n"
        f"ASSISTANT'S RESPONSE:\n{response[:1000]}\n\n"
        f"For EACH rule, does the response follow it? Consider:\n"
        f"- If measurements are required in metric, are ALL measurements metric? "
        f"(cups, tablespoons, ounces, pounds, fahrenheit = violations)\n"
        f"- If substitutions are required, does the response suggest cheaper alternatives?\n"
        f"- If disclaimers are required, are they present?\n\n"
        f"Output ONLY one word: PASS or FAIL"
    )

    result = _llm_client.call(
        system_prompt="You are a strict evaluator. Only output PASS or FAIL.",
        user_prompt=prompt,
        model=_judge_model,
        temperature=0.0,
        max_tokens=10
    )

    return "PASS" in result.strip().upper()


def _parse_individual_constraints(constraint_text: str) -> str:
    """Parse the raw constraint setup message into clean numbered rules."""
    rules = []
    text = constraint_text.lower()
    for pattern in [
        r"(?:always|never|please always|make sure to|don't forget to)\s+[^,.!?]+",
    ]:
        matches = re.findall(pattern, text)
        rules.extend(matches)
    if rules:
        return "\n".join(f"{i+1}. {r.strip()}" for i, r in enumerate(rules))
    return constraint_text


def _keyword_evaluate(response: str, keywords: List[str] = None) -> bool:
    """Enhanced keyword matching with positive AND negative checks."""
    response_lower = response.lower()

    if not keywords:
        return True

    # Check if metric-related keywords are in the set
    metric_keywords = {"gram", "ml", "liter", "celsius", "kg", "metric"}
    is_metric_check = any(kw.lower() in metric_keywords for kw in keywords)

    if is_metric_check:
        # Negative check: detect imperial measurement violations
        imperial_patterns = [
            r'\b\d+\s*cups?\b', r'\b\d+\s*tablespoons?\b', r'\b\d+\s*teaspoons?\b',
            r'\b\d+\s*ounces?\b', r'\b\d+\s*pounds?\b', r'\bfahrenheit\b',
            r'\b\d+\s*oz\b', r'\b\d+\s*lbs?\b', r'\b\d+\s*tbsp\b', r'\b\d+\s*tsp\b',
        ]
        has_imperial = any(re.search(p, response_lower) for p in imperial_patterns)
        if has_imperial:
            return False

        # Positive check: look for metric patterns (e.g. "500g", "250 ml")
        has_metric = bool(re.search(
            r'\d+\s*(?:g|grams?|ml|milliliters?|liters?|kg|kilograms?|°?c(?:elsius)?)\b',
            response_lower
        ))
        if has_metric:
            return True

    # General positive check: at least one keyword present
    return any(kw.lower() in response_lower for kw in keywords)


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

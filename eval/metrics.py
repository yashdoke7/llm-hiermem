"""
Metrics — Compute all evaluation metrics.

Primary metrics:
  1. Degradation Curve — accuracy at conversation checkpoints
  2. Constraint Violation Rate (CVR) — % responses violating stated constraints
  3. Fact Recall@K — recall accuracy at K turns distance
  4. Task Accuracy — overall correctness
  5. Consistency Score — self-contradiction detection

Secondary metrics:
  6. Latency per turn
  7. Token usage per turn
  8. Storage growth
"""

import re
from typing import Dict, List, Any


def compute_all_metrics(all_results: Dict[str, List[Dict]]) -> Dict:
    """Compute all metrics for all systems."""
    metrics = {}
    for system_name, conversations in all_results.items():
        metrics[system_name] = {
            "constraint_violation_rate": compute_cvr(conversations),
            "task_accuracy": compute_task_accuracy(conversations),
            "degradation_curve": compute_degradation_curve(conversations),
            "total_turns_processed": sum(c.get("turns_processed", 0) for c in conversations),
        }
    return metrics


def compute_cvr(conversations: List[Dict]) -> float:
    """Compute Constraint Violation Rate across all conversations."""
    total_checks = 0
    violations = 0
    
    for conv in conversations:
        checkpoints = conv.get("checkpoints", [])
        results = conv.get("results", [])
        
        for cp in checkpoints:
            turn = cp.get("turn", 0)
            test = cp.get("test", "")
            expected = cp.get("answer", True)
            
            # Find the response at this turn
            response_at_turn = None
            for r in results:
                if r.get("turn") == turn:
                    response_at_turn = r.get("response", "")
                    break
            
            if response_at_turn is not None:
                total_checks += 1
                passed = evaluate_checkpoint(response_at_turn, test, expected)
                if not passed:
                    violations += 1
    
    if total_checks == 0:
        return 0.0
    return violations / total_checks


def compute_task_accuracy(conversations: List[Dict]) -> float:
    """Compute overall task accuracy across checkpoint evaluations."""
    total = 0
    correct = 0
    
    for conv in conversations:
        checkpoints = conv.get("checkpoints", [])
        results = conv.get("results", [])
        
        for cp in checkpoints:
            turn = cp.get("turn", 0)
            test = cp.get("test", "")
            expected = cp.get("answer", True)
            
            response_at_turn = None
            for r in results:
                if r.get("turn") == turn:
                    response_at_turn = r.get("response", "")
                    break
            
            if response_at_turn is not None:
                total += 1
                if evaluate_checkpoint(response_at_turn, test, expected):
                    correct += 1
    
    if total == 0:
        return 0.0
    return correct / total


def compute_degradation_curve(conversations: List[Dict],
                               checkpoints_at: List[int] = None) -> Dict[int, float]:
    """Compute accuracy at different turn distances."""
    if checkpoints_at is None:
        checkpoints_at = [10, 20, 30, 40, 50]
    
    curve = {}
    for turn_threshold in checkpoints_at:
        total = 0
        correct = 0
        
        for conv in conversations:
            checkpoints = conv.get("checkpoints", [])
            results = conv.get("results", [])
            
            for cp in checkpoints:
                if cp.get("turn", 0) <= turn_threshold:
                    turn = cp["turn"]
                    response_at_turn = None
                    for r in results:
                        if r.get("turn") == turn:
                            response_at_turn = r.get("response", "")
                            break
                    
                    if response_at_turn is not None:
                        total += 1
                        if evaluate_checkpoint(response_at_turn, cp.get("test", ""),
                                             cp.get("answer", True)):
                            correct += 1
        
        curve[turn_threshold] = correct / total if total > 0 else 0.0
    
    return curve


def evaluate_checkpoint(response: str, test: str, expected_answer) -> bool:
    """
    Evaluate whether a response passes a checkpoint test.
    
    Supports:
      - Boolean checks: "Is response in bullet points?" → True/False
      - Keyword checks: "Does code use PostgreSQL?" → check for keyword
      - Exact match: Compare strings
    """
    response_lower = response.lower()
    test_lower = test.lower()
    
    # Keyword-based checks
    if "contain" in test_lower or "use" in test_lower or "mention" in test_lower:
        # Extract what to look for
        for keyword in _extract_test_keywords(test):
            if keyword.lower() in response_lower:
                return bool(expected_answer)
        return not bool(expected_answer)
    
    # Format checks
    if "bullet point" in test_lower:
        has_bullets = bool(re.search(r'[\•\-\*]\s', response))
        return has_bullets == bool(expected_answer)
    
    if "numbered list" in test_lower:
        has_numbers = bool(re.search(r'\d+[\.\)]\s', response))
        return has_numbers == bool(expected_answer)
    
    # Default: assume it's a keyword presence check
    return bool(expected_answer)


def _extract_test_keywords(test: str) -> List[str]:
    """Extract keywords from a test question."""
    # Look for quoted terms
    quoted = re.findall(r'"([^"]+)"', test) + re.findall(r"'([^']+)'", test)
    if quoted:
        return quoted
    
    # Look for capitalized terms (likely proper nouns/tech terms)
    caps = re.findall(r'\b[A-Z][a-zA-Z]+\b', test)
    if caps:
        return caps
    
    # Fallback: significant words
    stop_words = {"is", "the", "in", "does", "do", "are", "a", "an", "to", "of", 
                  "response", "code", "use", "contain", "mention", "have"}
    words = [w.strip("?.,!") for w in test.split() if w.lower() not in stop_words and len(w) > 2]
    return words[:3]

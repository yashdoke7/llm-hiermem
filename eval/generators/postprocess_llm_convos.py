"""
Post-process LLM-generated conversation files to add benchmark metadata.

LLM-generated files (from ChatGPT, Gemini, Claude) contain natural conversations
with type annotations (constraint_setup, checkpoint, etc.) in the turns but lack
the top-level metadata that run_benchmark.py and metrics.py need:
  - constraints list
  - checkpoints array with keywords, constraint_tested, test, answer
  - num_turns, num_checkpoints, domain_name

This script reads each file, extracts that info from the turns, and writes
back the enriched version.

Usage:
    python -m eval.generators.postprocess_llm_convos
    python -m eval.generators.postprocess_llm_convos --dir eval/datasets/constraint_tracking/llm_generated
    python -m eval.generators.postprocess_llm_convos --file eval/datasets/constraint_tracking/llm_generated/chatgpt_llm_se1.json
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Keyword bank — maps constraint phrases to evaluation keywords.
# The evaluator (metrics.py) checks if ANY keyword appears in the response.
# ---------------------------------------------------------------------------
CONSTRAINT_KEYWORDS: Dict[str, List[str]] = {
    # Code style constraints
    "type hint": [
        ":", "->", "int", "str", "float", "bool", "List[", "Dict[",
        "Optional[", "Tuple[", "type hint", "typing", "annotat",
    ],
    "try/except": [
        "try:", "except", "try/except", "Exception", "error handling",
        "raise", "catch",
    ],
    "error handling": [
        "try:", "except", "try/except", "Exception", "error handling",
        "raise", "catch", "error",
    ],
    # Knowledge constraints
    "time complexity": [
        "O(", "complexity", "time complexity", "space complexity",
        "Big O", "linear", "quadratic", "logarithmic", "constant time",
        "exponential", "O(n", "O(1", "O(log",
    ],
    "test": [
        "test", "pytest", "unittest", "unit test", "assert", "testing",
        "test case", "test suite", "mock", "coverage",
    ],
    "write test": [
        "test", "pytest", "unittest", "unit test", "assert", "testing",
        "test case", "test suite", "mock", "coverage",
    ],
    # Format constraints
    "bullet point": ["•", "-", "*", "bullet"],
    "numbered list": ["1.", "2.", "3.", "numbered"],
    "markdown": ["```", "**", "##", "markdown"],
    # Medical
    "cite source": ["study", "journal", "research", "source", "evidence", "according"],
    "side effect": ["side effect", "adverse", "risk", "caution", "warning"],
    "drug interaction": ["interaction", "contraindic", "combine", "concurrent"],
    # Legal
    "jurisdiction": ["jurisdiction", "state law", "federal", "varies by"],
    "disclaimer": ["not legal advice", "consult", "disclaimer", "attorney"],
    # Cooking
    "metric": ["gram", "ml", "liter", "celsius", "kg", "metric"],
    "substitut": ["substitut", "alternative", "replace", "instead of"],
    # Finance
    "risk": ["risk", "volatil", "diversif", "loss", "downside"],
    "disclaim": ["not financial advice", "consult", "professional", "disclaimer"],
    # General
    "example": ["example", "for instance", "e.g.", "such as", "like"],
    "analogy": ["analogy", "like", "think of it as", "imagine", "similar to"],
    "step by step": ["step 1", "step 2", "first", "then", "next", "finally"],
    "pros and cons": ["pro", "con", "advantage", "disadvantage", "tradeoff"],
}

DOMAIN_NAMES = {
    "software_engineering": "Software Engineering",
    "medical_consultation": "Medical Consultation",
    "legal_advisory": "Legal Advisory",
    "academic_tutoring": "Academic Tutoring",
    "customer_support": "Customer Support",
    "creative_writing": "Creative Writing",
    "data_analysis": "Data Analysis",
    "project_management": "Project Management",
    "cooking_recipes": "Cooking & Recipes",
    "personal_finance": "Personal Finance",
}


def extract_constraints(turns: List[Dict]) -> List[str]:
    """Extract constraint text from constraint_setup turns."""
    constraints = []
    for t in turns:
        if t.get("type") == "constraint_setup" and t.get("role") == "user":
            constraints.append(t["text"])
    return constraints


def match_keywords_for_constraint(constraint_text: str) -> List[str]:
    """Find the best keyword list for a constraint based on text matching."""
    text_lower = constraint_text.lower()
    best_match = None
    best_score = 0

    for pattern, keywords in CONSTRAINT_KEYWORDS.items():
        if pattern.lower() in text_lower:
            # Prefer longer pattern matches (more specific)
            score = len(pattern)
            if score > best_score:
                best_score = score
                best_match = keywords

    if best_match:
        return best_match

    # Fallback: extract significant words from the constraint
    stop = {"always", "make", "sure", "you", "the", "and", "also", "please",
            "pls", "can", "use", "every", "time", "whenever", "include",
            "proper", "with", "for", "from", "that", "this", "any", "all",
            "when", "show", "code", "want", "like", "just", "dont", "im",
            "its", "your", "about", "into", "each", "should", "will", "are"}
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    keywords = [w for w in words if w not in stop][:5]
    return keywords if keywords else ["constraint"]


def parse_constraint_phrases(constraint_text: str) -> List[Tuple[str, List[str]]]:
    """Parse a constraint_setup message to extract individual constraint phrases + keywords.

    A single constraint_setup turn often contains multiple constraints, e.g.:
    "always use type hints, and also pls include try/except handling"
    """
    text_lower = constraint_text.lower()
    found: List[Tuple[str, List[str]]] = []

    for pattern, keywords in CONSTRAINT_KEYWORDS.items():
        if pattern.lower() in text_lower:
            found.append((pattern, keywords))

    if not found:
        # Treat the whole text as one constraint
        kw = match_keywords_for_constraint(constraint_text)
        found.append((constraint_text[:80], kw))

    return found


def find_checkpoint_turns(turns: List[Dict]) -> List[int]:
    """Find turn numbers that are checkpoints."""
    return [t["turn"] for t in turns
            if t.get("type") == "checkpoint" and t.get("role") == "user"]


def build_checkpoints(
    constraint_phrases: List[Tuple[str, List[str]]],
    checkpoint_turns: List[int],
    original_constraints: List[str],
) -> List[Dict]:
    """Build the checkpoints metadata array.

    Each checkpoint turn tests each constraint, so total checkpoints =
    len(checkpoint_turns) × len(constraint_phrases).
    """
    checkpoints = []
    for turn_num in checkpoint_turns:
        for phrase, keywords in constraint_phrases:
            # Find which original constraint text contains this phrase
            constraint_text = phrase
            for orig in original_constraints:
                if phrase.lower() in orig.lower():
                    constraint_text = orig
                    break

            checkpoints.append({
                "turn": turn_num,
                "constraint_tested": constraint_text,
                "test": f"Does response follow: '{constraint_text}'?",
                "answer": True,
                "keywords": keywords,
            })
    return checkpoints


def count_user_turns(turns: List[Dict]) -> int:
    """Count unique user turns (each turn number appears once for user + once for assistant)."""
    return len(set(t["turn"] for t in turns if t.get("role") == "user"))


def postprocess_file(filepath: Path, dry_run: bool = False) -> Dict:
    """Post-process a single LLM-generated conversation file."""
    data = json.loads(filepath.read_text(encoding="utf-8"))

    # Handle both single-object and array formats
    if isinstance(data, list):
        conversations = data
    else:
        conversations = [data]

    processed = []
    for i, conv in enumerate(conversations):
        turns = conv.get("turns", [])
        domain = conv.get("domain", "unknown")

        # Extract constraints from turn content
        raw_constraints = extract_constraints(turns)
        constraint_phrases = []
        for c in raw_constraints:
            constraint_phrases.extend(parse_constraint_phrases(c))

        # Deduplicate constraint phrases by pattern name
        seen = set()
        unique_phrases = []
        for phrase, kw in constraint_phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append((phrase, kw))
        constraint_phrases = unique_phrases

        # Find checkpoint turns
        checkpoint_turns = find_checkpoint_turns(turns)

        # Build metadata
        checkpoints = build_checkpoints(
            constraint_phrases, checkpoint_turns, raw_constraints
        )

        # Fix conversation_id to be unique (use filename stem + index)
        base_id = conv.get("conversation_id", f"llm_gen_{domain}")
        new_id = f"{filepath.stem}_{i}" if len(conversations) > 1 else filepath.stem

        enriched = {
            "conversation_id": new_id,
            "domain": domain,
            "domain_name": DOMAIN_NAMES.get(domain, domain.replace("_", " ").title()),
            "source": filepath.stem.split("_")[0],  # chatgpt, gemini, claude
            "constraints": raw_constraints,
            "num_turns": count_user_turns(turns),
            "num_checkpoints": len(checkpoints),
            "checkpoints": checkpoints,
            "turns": turns,
        }
        processed.append(enriched)

        # Report
        print(f"  {filepath.name}: {enriched['num_turns']} turns, "
              f"{len(constraint_phrases)} constraints, "
              f"{len(checkpoint_turns)} checkpoint turns → "
              f"{len(checkpoints)} checkpoint evaluations")
        for phrase, kw in constraint_phrases:
            print(f"    constraint: '{phrase}' → {len(kw)} keywords")

    if not dry_run:
        # Write back — always as array for consistency with synthetic files
        filepath.write_text(json.dumps(processed, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")

    return processed


def postprocess_directory(dirpath: Path, dry_run: bool = False):
    """Post-process all JSON files in a directory."""
    files = sorted(dirpath.glob("*.json"))
    if not files:
        print(f"No JSON files found in {dirpath}")
        return

    print(f"Processing {len(files)} files in {dirpath}")
    total_convos = 0
    total_checkpoints = 0

    for f in files:
        result = postprocess_file(f, dry_run)
        for conv in result:
            total_convos += 1
            total_checkpoints += conv["num_checkpoints"]

    print(f"\nTotal: {total_convos} conversations, {total_checkpoints} checkpoint evaluations")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process LLM-generated conversations to add benchmark metadata"
    )
    parser.add_argument("--file", type=str, help="Process a single file")
    parser.add_argument("--dir", type=str, help="Process all JSON files in directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze without writing changes")
    parser.add_argument("--all", action="store_true",
                        help="Process all llm_generated/ dirs across all benchmarks")
    args = parser.parse_args()

    base = Path(__file__).parent.parent / "datasets"

    if args.file:
        postprocess_file(Path(args.file), args.dry_run)
    elif args.dir:
        postprocess_directory(Path(args.dir), args.dry_run)
    elif args.all:
        for benchmark_dir in sorted(base.iterdir()):
            llm_dir = benchmark_dir / "llm_generated"
            if llm_dir.is_dir() and list(llm_dir.glob("*.json")):
                print(f"\n{'='*60}")
                print(f"Benchmark: {benchmark_dir.name}")
                print(f"{'='*60}")
                postprocess_directory(llm_dir, args.dry_run)
    else:
        # Default: process constraint_tracking/llm_generated
        default_dir = base / "constraint_tracking" / "llm_generated"
        if default_dir.is_dir():
            postprocess_directory(default_dir, args.dry_run)
        else:
            print(f"No llm_generated/ directory found at {default_dir}")
            print("Use --file, --dir, or --all to specify what to process")


if __name__ == "__main__":
    main()

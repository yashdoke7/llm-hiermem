"""
LLM-Powered Natural Conversation Generator

Uses an LLM (Ollama, or manually via ChatGPT/Gemini/Claude) to generate
realistic, human-like conversations for HierMem evaluation.

Usage:
  # With Ollama (automated):
  python -m eval.generators.generate_llm_convos --model ollama/llama3.1:8b --domain software_engineering

  # Manual mode: prints the prompt for you to paste into ChatGPT/Gemini/Claude
  python -m eval.generators.generate_llm_convos --manual --domain medical_consultation
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Domain scenario templates — each gives the LLM a setup to role-play
SCENARIOS = {
    "software_engineering": [
        {
            "scenario": "A junior developer is building their first production REST API in Python and asking for guidance over many turns.",
            "constraints": [
                "always use type hints in code examples",
                "always include error handling (try/except) in code",
            ],
            "keywords_map": {
                "always use type hints in code examples": [": str", ": int", ": float", ": bool", "-> ", ": List", ": Dict", ": Optional"],
                "always include error handling (try/except) in code": ["try", "except", "raise", "Error", "Exception"],
            },
        },
        {
            "scenario": "A developer is refactoring a legacy codebase and needs help with design patterns, testing, and optimization.",
            "constraints": [
                "always suggest writing tests when implementing new functions",
                "always explain time complexity of algorithms",
            ],
            "keywords_map": {
                "always suggest writing tests when implementing new functions": ["test", "unittest", "pytest", "assert", "test_", "mock"],
                "always explain time complexity of algorithms": ["O(", "complexity", "time complexity", "Big O", "linear", "logarithmic"],
            },
        },
    ],
    "medical_consultation": [
        {
            "scenario": "A 40-year-old has been diagnosed with Type 2 diabetes and is asking about management, diet, exercise, and medication over multiple sessions.",
            "constraints": [
                "always mention potential side effects when discussing any medication",
                "always recommend consulting a healthcare professional before starting treatment",
            ],
            "keywords_map": {
                "always mention potential side effects when discussing any medication": ["side effect", "adverse", "may cause", "risk of", "nausea", "caution"],
                "always recommend consulting a healthcare professional before starting treatment": ["consult", "doctor", "physician", "healthcare professional", "medical professional"],
            },
        },
        {
            "scenario": "A parent asking about their child's recurring health issues — allergies, fevers, nutrition questions — across many conversations.",
            "constraints": [
                "always ask about existing allergies before suggesting any medication",
                "always recommend consulting a healthcare professional before starting treatment",
            ],
            "keywords_map": {
                "always ask about existing allergies before suggesting any medication": ["allerg", "allergic", "sensitivity", "reaction to", "any allergies"],
                "always recommend consulting a healthcare professional before starting treatment": ["consult", "doctor", "physician", "healthcare professional"],
            },
        },
    ],
    "legal_advisory": [
        {
            "scenario": "Someone starting a small business asking about legal structure, contracts, liability, hiring — over many turns.",
            "constraints": [
                "always include a disclaimer that this is general information, not legal advice",
                "always mention that laws vary by jurisdiction",
            ],
            "keywords_map": {
                "always include a disclaimer that this is general information, not legal advice": ["not legal advice", "disclaimer", "consult an attorney", "general information"],
                "always mention that laws vary by jurisdiction": ["jurisdiction", "vary by state", "depends on your location", "state law", "local law"],
            },
        },
    ],
    "academic_tutoring": [
        {
            "scenario": "A college student learning calculus from scratch — limits, derivatives, integrals — over a tutoring-style multi-turn conversation.",
            "constraints": [
                "always show step-by-step work when solving problems",
                "always provide a practice problem at the end of explanations",
            ],
            "keywords_map": {
                "always show step-by-step work when solving problems": ["step 1", "step 2", "first,", "next,", "then,", "therefore", "step-by-step"],
                "always provide a practice problem at the end of explanations": ["practice", "try this", "exercise", "your turn", "try solving"],
            },
        },
        {
            "scenario": "A student learning about machine learning concepts — from linear regression to neural networks — in a progressive tutoring style.",
            "constraints": [
                "always use real-world analogies when explaining abstract concepts",
                "always show step-by-step work when solving problems",
            ],
            "keywords_map": {
                "always use real-world analogies when explaining abstract concepts": ["like", "similar to", "analogy", "imagine", "think of it as", "just like"],
                "always show step-by-step work when solving problems": ["step 1", "step 2", "first,", "next,", "therefore", "step-by-step"],
            },
        },
    ],
    "customer_support": [
        {
            "scenario": "A frustrated customer dealing with recurring internet connectivity issues, billing problems, and trying to get proper support across multiple interactions.",
            "constraints": [
                "always acknowledge the customer's frustration before providing solutions",
                "always offer to escalate if the issue isn't resolved",
            ],
            "keywords_map": {
                "always acknowledge the customer's frustration before providing solutions": ["understand", "sorry", "frustrat", "apologize", "inconvenien"],
                "always offer to escalate if the issue isn't resolved": ["escalat", "supervisor", "specialist", "senior", "manager"],
            },
        },
    ],
    "creative_writing": [
        {
            "scenario": "A writer collaborating on a mystery novel — developing characters, plot, scenes — over many turns of creative back-and-forth.",
            "constraints": [
                "always use vivid sensory details in story descriptions",
                "always include dialogue between characters in story passages",
            ],
            "keywords_map": {
                "always use vivid sensory details in story descriptions": ["smell", "sound", "taste", "hear", "felt", "scent", "aroma", "glow", "warm", "cold"],
                "always include dialogue between characters in story passages": ["said", "asked", "replied", "whispered", "\"", "told", "answered"],
            },
        },
    ],
    "data_analysis": [
        {
            "scenario": "A product analyst running their first A/B test and cohort analysis — from design to interpretation to presentation.",
            "constraints": [
                "always state the sample size when presenting statistical findings",
                "always mention potential biases or limitations when interpreting data",
            ],
            "keywords_map": {
                "always state the sample size when presenting statistical findings": ["n =", "n=", "sample size", "sample of", "observations", "data points"],
                "always mention potential biases or limitations when interpreting data": ["bias", "limitation", "caveat", "however", "important to note", "correlation"],
            },
        },
    ],
    "project_management": [
        {
            "scenario": "A new engineering manager planning a product launch — sprint planning, risk management, team scaling — over many turns.",
            "constraints": [
                "always identify risks and mitigation strategies when proposing plans",
                "always break down work into measurable milestones",
            ],
            "keywords_map": {
                "always identify risks and mitigation strategies when proposing plans": ["risk", "mitigat", "contingenc", "fallback", "what if"],
                "always break down work into measurable milestones": ["milestone", "deliverable", "phase", "sprint", "week 1", "deadline"],
            },
        },
    ],
    "cooking_recipes": [
        {
            "scenario": "A beginner learning to bake — starting from cookies, moving to bread, then pastries — asking questions over many turns.",
            "constraints": [
                "always mention cooking temperatures in both Celsius and Fahrenheit",
                "always suggest allergen substitutions for each recipe",
            ],
            "keywords_map": {
                "always mention cooking temperatures in both Celsius and Fahrenheit": ["°C", "°F", "Celsius", "Fahrenheit", "350F", "180C"],
                "always suggest allergen substitutions for each recipe": ["substitut", "alternative", "replace", "instead of", "dairy-free", "gluten-free"],
            },
        },
    ],
    "personal_finance": [
        {
            "scenario": "A 28-year-old starting their financial journey — investing basics, debt management, retirement planning — over many turns.",
            "constraints": [
                "always remind that past performance doesn't guarantee future returns",
                "always recommend consulting a certified financial advisor",
            ],
            "keywords_map": {
                "always remind that past performance doesn't guarantee future returns": ["past performance", "no guarantee", "doesn't guarantee", "risk", "historical"],
                "always recommend consulting a certified financial advisor": ["financial advisor", "certified", "professional advice", "CFP", "financial professional"],
            },
        },
    ],
}


def build_generation_prompt(
    domain: str,
    scenario_idx: int = 0,
    num_turns: int = 50,
    checkpoint_turns: List[int] = None,
) -> str:
    """Build a prompt that instructs an LLM to generate a natural conversation."""

    scenarios = SCENARIOS.get(domain, [])
    if not scenarios:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(SCENARIOS.keys())}")
    scenario = scenarios[scenario_idx % len(scenarios)]

    if checkpoint_turns is None:
        checkpoint_turns = [8, 15, 25, 35, 45]

    constraints_text = "\n".join(f"  - {c}" for c in scenario["constraints"])
    checkpoints_text = ", ".join(str(t) for t in checkpoint_turns)

    prompt = f"""Generate a realistic {num_turns}-turn conversation between a Human and an AI Assistant.

SCENARIO: {scenario['scenario']}

CONSTRAINTS (stated by the human at the start, in natural casual language — NOT as a numbered list):
{constraints_text}

RULES FOR GENERATION:
1. The human states their preferences/constraints naturally in turn 1 (like a real person would — casual, not robotic)
2. The conversation flows naturally — each turn builds on the previous one
3. The human asks follow-up questions, changes subtopics within the domain, goes on tangents, comes back
4. The human uses casual language with typos, abbreviations, incomplete sentences sometimes
5. The AI gives helpful, detailed responses (100-300 words each)
6. At turns [{checkpoints_text}], the human asks something that should trigger the constraint — but does NOT mention the constraint explicitly or hint at it
7. The AI's response at checkpoint turns MUST follow the constraints (this is what we test)
8. Non-checkpoint turns should be natural domain conversation — the AI may or may not follow constraints

OUTPUT FORMAT (valid JSON):
{{
  "conversation_id": "llm_gen_{domain}_001",
  "domain": "{domain}",
  "turns": [
    {{"turn": 1, "role": "user", "text": "...", "type": "constraint_setup"}},
    {{"turn": 1, "role": "assistant", "text": "...", "type": "ack"}},
    {{"turn": 2, "role": "user", "text": "...", "type": "topic"}},
    {{"turn": 2, "role": "assistant", "text": "...", "type": "topic_response"}},
    ...
    {{"turn": 8, "role": "user", "text": "...", "type": "checkpoint"}},
    {{"turn": 8, "role": "assistant", "text": "...", "type": "checkpoint_response"}},
    ...
  ]
}}

Mark checkpoint turns with type "checkpoint" / "checkpoint_response".
Mark the first turn as "constraint_setup" / "ack".
All other turns are "topic" / "topic_response".

Generate the full {num_turns}-turn conversation now. Output ONLY valid JSON, no other text."""

    return prompt


def build_checkpoint_metadata(
    domain: str,
    scenario_idx: int = 0,
    checkpoint_turns: List[int] = None,
) -> List[Dict]:
    """Build checkpoint evaluation metadata for a generated conversation."""
    scenarios = SCENARIOS[domain]
    scenario = scenarios[scenario_idx % len(scenarios)]

    if checkpoint_turns is None:
        checkpoint_turns = [8, 15, 25, 35, 45]

    checkpoints = []
    for i, turn in enumerate(checkpoint_turns):
        constraint = scenario["constraints"][i % len(scenario["constraints"])]
        keywords = scenario["keywords_map"][constraint]
        checkpoints.append({
            "turn": turn,
            "constraint_tested": constraint,
            "test": f"Does response follow: '{constraint}'?",
            "answer": True,
            "keywords": keywords,
        })
    return checkpoints


def parse_llm_output(
    raw_json: str,
    domain: str,
    scenario_idx: int = 0,
    checkpoint_turns: List[int] = None,
) -> Dict:
    """Parse LLM output and add checkpoint metadata."""
    # Try to extract JSON from the output
    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]

    conv = json.loads(raw_json)

    # Add checkpoint metadata
    conv["checkpoints"] = build_checkpoint_metadata(domain, scenario_idx, checkpoint_turns)
    conv["domain"] = domain
    conv["domain_name"] = SCENARIOS[domain][scenario_idx % len(SCENARIOS[domain])].get(
        "scenario", domain
    )[:50]
    conv["constraints"] = SCENARIOS[domain][scenario_idx % len(SCENARIOS[domain])]["constraints"]
    conv["num_turns"] = max(t["turn"] for t in conv.get("turns", [{"turn": 0}]))
    conv["num_checkpoints"] = len(conv["checkpoints"])

    return conv


def generate_with_ollama(
    domain: str,
    scenario_idx: int = 0,
    num_turns: int = 50,
    model: str = "ollama/llama3.1:8b",
) -> Optional[Dict]:
    """Generate a conversation using Ollama."""
    from hiermem.llm.client import LLMClient

    prompt = build_generation_prompt(domain, scenario_idx, num_turns)
    client = LLMClient(provider="ollama")

    print(f"  Generating {domain} conversation via {model}...")
    try:
        response = client.call(
            system_prompt="You are a conversation generator. Output only valid JSON.",
            user_prompt=prompt,
            model=model,
            temperature=0.7,
        )
        return parse_llm_output(response, domain, scenario_idx)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate natural conversations using LLMs"
    )
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain to generate (default: all)")
    parser.add_argument("--model", type=str, default="ollama/llama3.1:8b",
                        help="Ollama model to use")
    parser.add_argument("--num-per-domain", type=int, default=2,
                        help="Conversations per domain")
    parser.add_argument("--turns", type=int, default=30,
                        help="Turns per conversation")
    parser.add_argument("--manual", action="store_true",
                        help="Print prompts for manual generation (ChatGPT/Gemini/Claude)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    args = parser.parse_args()

    domains = [args.domain] if args.domain else list(SCENARIOS.keys())
    output_path = Path(args.output) if args.output else (
        Path(__file__).parent.parent / "datasets" / "constraint_tracking" / "llm_generated"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    if args.manual:
        # Print prompts for manual copy-paste
        for domain in domains:
            scenarios = SCENARIOS[domain]
            for idx in range(min(args.num_per_domain, len(scenarios))):
                prompt = build_generation_prompt(domain, idx, args.turns)
                print(f"\n{'='*70}")
                print(f"DOMAIN: {domain} | Scenario {idx+1}/{len(scenarios)}")
                print(f"{'='*70}")
                print(prompt)
                print(f"\n--- Save output as: {output_path / domain}.json ---\n")
        return

    # Automated generation with Ollama
    for domain in domains:
        scenarios = SCENARIOS[domain]
        convos = []
        for idx in range(min(args.num_per_domain, len(scenarios))):
            conv = generate_with_ollama(domain, idx, args.turns, args.model)
            if conv:
                convos.append(conv)

        if convos:
            out_file = output_path / f"{domain}.json"
            out_file.write_text(json.dumps(convos, indent=2))
            print(f"  Saved {len(convos)} conversations → {out_file}")


if __name__ == "__main__":
    main()

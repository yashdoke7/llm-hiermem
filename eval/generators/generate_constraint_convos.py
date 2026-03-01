"""
Generate synthetic constraint-tracking conversations.

Creates conversations where:
1. Constraints/rules are stated early
2. Conversation proceeds with filler turns
3. Checkpoints test whether constraints are still remembered
"""

import json
import random
import argparse
from pathlib import Path

# Templates for constraints
CONSTRAINTS = [
    ("Never use global variables in code", ["global"]),
    ("Always respond in bullet points", ["bullet"]),
    ("Use PostgreSQL, never MySQL", ["PostgreSQL", "MySQL"]),
    ("Keep all responses under 100 words", ["100 words"]),
    ("Never suggest deleting user data without confirmation", ["delete", "user data"]),
    ("Always include error handling in code examples", ["try", "except", "error"]),
    ("Use type hints in all Python code", ["type hint", ": str", ": int", "-> "]),
    ("Never hardcode API keys", ["API key", "hardcode"]),
    ("Always use async/await for I/O operations", ["async", "await"]),
    ("Format all SQL queries in uppercase keywords", ["SELECT", "FROM", "WHERE"]),
    ("Never recommend pip install without a virtual environment warning", ["pip install", "venv"]),
    ("Always suggest tests when writing new functions", ["test", "unittest", "pytest"]),
    ("Use pathlib instead of os.path", ["pathlib", "Path"]),
    ("Always explain time complexity of algorithms", ["O(n)", "complexity", "Big O"]),
    ("Never use print() for logging in production code", ["logging", "print"]),
]

# Filler conversation topics
FILLER_TOPICS = [
    ("How do I read a CSV file in Python?", "You can use pandas..."),
    ("Explain list comprehensions", "List comprehensions provide..."),
    ("What's the difference between a list and tuple?", "Lists are mutable..."),
    ("How do I handle exceptions?", "Use try/except blocks..."),
    ("Explain decorators in Python", "Decorators are functions..."),
    ("What is a context manager?", "Context managers handle..."),
    ("How do I use regular expressions?", "The re module provides..."),
    ("Explain generators", "Generators use yield..."),
    ("What is the GIL?", "The Global Interpreter Lock..."),
    ("How do I profile Python code?", "Use cProfile or..."),
    ("What are dataclasses?", "Dataclasses reduce boilerplate..."),
    ("How do I use asyncio?", "asyncio provides event loops..."),
    ("Explain metaclasses", "Metaclasses define class behavior..."),
    ("How do I optimize memory usage?", "Use generators, __slots__..."),
    ("What is dependency injection?", "DI is a design pattern..."),
]

# Checkpoint test templates
CHECKPOINT_TESTS = {
    "recall": "What constraints did the user set at the beginning?",
    "apply_code": "Write a Python function to {task}",
    "apply_format": "Explain {topic} in detail",
}


def generate_conversation(num_turns: int = 50, num_constraints: int = 2,
                          num_checkpoints: int = 5, seed: int = None) -> dict:
    """Generate a single synthetic conversation."""
    if seed is not None:
        random.seed(seed)
    
    # Pick constraints
    selected = random.sample(CONSTRAINTS, min(num_constraints, len(CONSTRAINTS)))
    
    turns = []
    checkpoints = []
    turn_num = 0
    
    # Turn 1: State constraints
    constraint_text = "Important rules for our conversation:\n"
    for i, (text, _) in enumerate(selected):
        constraint_text += f"{i+1}. {text}\n"
    constraint_text += "Please follow these rules in ALL your responses."
    
    turn_num += 1
    turns.append({
        "turn": turn_num,
        "role": "user",
        "text": constraint_text,
        "type": "constraint_setup"
    })
    turns.append({
        "turn": turn_num,
        "role": "assistant",
        "text": "Understood! I will follow these rules throughout our conversation.",
        "type": "ack"
    })
    
    # Generate filler + checkpoint turns
    checkpoint_turns = sorted(random.sample(
        range(5, num_turns, 3), min(num_checkpoints, (num_turns - 5) // 3)
    ))
    
    filler_pool = list(FILLER_TOPICS) * 5  # repeat to have enough
    random.shuffle(filler_pool)
    filler_idx = 0
    
    for t in range(2, num_turns + 1):
        turn_num = t
        
        if t in checkpoint_turns:
            # Checkpoint: test constraint recall
            constraint_idx = random.randint(0, len(selected) - 1)
            text, keywords = selected[constraint_idx]
            
            test_prompt = f"Write a Python function to process user data. Remember to follow all our agreed rules."
            turns.append({"turn": turn_num, "role": "user", "text": test_prompt, "type": "checkpoint"})
            turns.append({"turn": turn_num, "role": "assistant", "text": "[RESPONSE_PLACEHOLDER]", "type": "checkpoint_response"})
            
            checkpoints.append({
                "turn": turn_num,
                "constraint_tested": text,
                "test": f"Does response contain or follow: '{text}'?",
                "answer": True,
                "keywords": keywords,
            })
        else:
            # Filler turn
            if filler_idx < len(filler_pool):
                user_msg, assistant_msg = filler_pool[filler_idx]
                filler_idx += 1
            else:
                user_msg = f"Tell me about topic {t}"
                assistant_msg = f"Here's information about topic {t}..."
            
            turns.append({"turn": turn_num, "role": "user", "text": user_msg, "type": "filler"})
            turns.append({"turn": turn_num, "role": "assistant", "text": assistant_msg, "type": "filler"})
    
    return {
        "conversation_id": f"constraint_{seed or random.randint(0, 99999):05d}",
        "constraints": [c[0] for c in selected],
        "num_turns": num_turns,
        "checkpoints": checkpoints,
        "turns": turns,
    }


def generate_dataset(num_conversations: int = 50, turns_per_convo: int = 50,
                     output_path: Path = None):
    """Generate the full constraint-tracking dataset."""
    output_path = output_path or Path(__file__).parent.parent / "datasets" / "constraint_tracking"
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    for i in range(num_conversations):
        conv = generate_conversation(
            num_turns=turns_per_convo,
            num_constraints=random.randint(1, 4),
            num_checkpoints=5,
            seed=i
        )
        dataset.append(conv)
    
    out_file = output_path / "conversations.json"
    out_file.write_text(json.dumps(dataset, indent=2))
    print(f"Generated {num_conversations} conversations → {out_file}")
    
    # Summary stats
    total_checkpoints = sum(len(c["checkpoints"]) for c in dataset)
    print(f"Total checkpoints: {total_checkpoints}")
    print(f"Avg checkpoints/convo: {total_checkpoints / num_conversations:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=50)
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    output = Path(args.output) if args.output else None
    generate_dataset(args.num_conversations, args.turns, output)

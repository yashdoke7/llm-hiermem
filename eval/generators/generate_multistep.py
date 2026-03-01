"""
Generate multi-step reasoning conversations.

Creates conversations where later turns depend on facts from earlier turns,
testing whether the system can chain information across long distances.
"""

import json
import random
import argparse
from pathlib import Path


REASONING_CHAINS = [
    {
        "setup_turns": [
            ("The server runs on Ubuntu 22.04.", "Noted: Ubuntu 22.04."),
            ("We use nginx as the reverse proxy.", "Got it: nginx reverse proxy."),
            ("The app runs on port 5000 internally.", "Internal port: 5000."),
        ],
        "question": "Write the nginx config to proxy requests to our app.",
        "required_facts": ["Ubuntu", "nginx", "5000"],
        "test": "Does the nginx config route to port 5000?"
    },
    {
        "setup_turns": [
            ("Our database is PostgreSQL 15.", "Noted."),
            ("The main table is called 'events' with columns: id, name, timestamp, user_id.", "Got it."),
            ("We need to partition by month on the timestamp column.", "Understood."),
        ],
        "question": "Write the SQL to create the partitioned events table.",
        "required_facts": ["PostgreSQL", "events", "timestamp", "partition"],
        "test": "Does SQL create partitioned events table with correct columns?"
    },
    {
        "setup_turns": [
            ("The project uses FastAPI.", "Noted: FastAPI."),
            ("Auth is handled via JWT tokens with RS256.", "Got it: JWT + RS256."),
            ("Tokens expire after 15 minutes.", "Token expiry: 15 min."),
            ("Refresh tokens last 7 days.", "Refresh: 7 days."),
        ],
        "question": "Implement the token generation and validation endpoints.",
        "required_facts": ["FastAPI", "JWT", "RS256", "15 minutes", "7 days"],
        "test": "Does implementation use FastAPI, JWT, RS256, 15min expiry?"
    },
    {
        "setup_turns": [
            ("We use Redis for caching with a 5-minute TTL.", "Noted."),
            ("The cache key format is 'user:{user_id}:profile'.", "Got it."),
            ("On cache miss, fetch from PostgreSQL.", "Understood."),
        ],
        "question": "Write the caching logic for user profile lookups.",
        "required_facts": ["Redis", "5-minute", "user:{user_id}:profile", "PostgreSQL"],
        "test": "Does caching use Redis with correct key format and TTL?"
    },
]

FILLER_TURNS = [
    ("What's a good IDE for Python?", "VS Code or PyCharm are popular."),
    ("How do I format code?", "Use black or autopep8."),
    ("What's PEP 8?", "Python's style guide."),
    ("Explain virtual environments.", "Isolated Python environments."),
    ("What is pip?", "Python's package manager."),
    ("How do I debug Python?", "Use pdb or IDE debugger."),
    ("What are f-strings?", "Formatted string literals in Python 3.6+."),
    ("How do I read JSON?", "Use json.load() or json.loads()."),
    ("What is WSGI?", "Web Server Gateway Interface."),
    ("Explain REST APIs.", "HTTP-based stateless interfaces."),
]


def generate_multistep_conversation(chain_idx: int = 0, filler_between: int = 5,
                                      total_turns: int = 50, seed: int = None) -> dict:
    """Generate a multi-step reasoning conversation."""
    if seed is not None:
        random.seed(seed)
    
    chain = REASONING_CHAINS[chain_idx % len(REASONING_CHAINS)]
    
    turns = []
    checkpoints = []
    fillers = list(FILLER_TURNS) * 10
    random.shuffle(fillers)
    filler_idx = 0
    turn_num = 0
    
    # Setup turns (early in conversation)
    for user_msg, asst_msg in chain["setup_turns"]:
        turn_num += 1
        turns.append({"turn": turn_num, "role": "user", "text": user_msg, "type": "setup"})
        turns.append({"turn": turn_num, "role": "assistant", "text": asst_msg, "type": "setup"})
    
    # Filler turns
    for _ in range(filler_between):
        turn_num += 1
        if filler_idx < len(fillers):
            u, a = fillers[filler_idx]
            filler_idx += 1
        else:
            u, a = f"Question {turn_num}", f"Answer {turn_num}"
        turns.append({"turn": turn_num, "role": "user", "text": u, "type": "filler"})
        turns.append({"turn": turn_num, "role": "assistant", "text": a, "type": "filler"})
    
    # Question turn
    turn_num += 1
    turns.append({"turn": turn_num, "role": "user", "text": chain["question"], "type": "test"})
    turns.append({"turn": turn_num, "role": "assistant", "text": "[RESPONSE_PLACEHOLDER]", "type": "test_response"})
    
    checkpoints.append({
        "turn": turn_num,
        "required_facts": chain["required_facts"],
        "test": chain["test"],
        "answer": True,
    })
    
    # More filler to pad to total_turns
    while turn_num < total_turns:
        turn_num += 1
        if filler_idx < len(fillers):
            u, a = fillers[filler_idx]
            filler_idx += 1
        else:
            u, a = f"Question {turn_num}", f"Answer {turn_num}"
        turns.append({"turn": turn_num, "role": "user", "text": u, "type": "filler"})
        turns.append({"turn": turn_num, "role": "assistant", "text": a, "type": "filler"})
    
    return {
        "conversation_id": f"multistep_{seed or random.randint(0, 99999):05d}",
        "chain_type": chain["test"],
        "num_turns": turn_num,
        "filler_between": filler_between,
        "checkpoints": checkpoints,
        "turns": turns,
    }


def generate_dataset(num_conversations: int = 40, output_path: Path = None):
    """Generate full multi-step dataset with varying gap sizes."""
    output_path = output_path or Path(__file__).parent.parent / "datasets" / "multistep_reasoning"
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    for i in range(num_conversations):
        gap = random.choice([5, 10, 15, 20, 30])
        conv = generate_multistep_conversation(
            chain_idx=i,
            filler_between=gap,
            total_turns=gap + 10,
            seed=i
        )
        dataset.append(conv)
    
    out_file = output_path / "conversations.json"
    out_file.write_text(json.dumps(dataset, indent=2))
    print(f"Generated {num_conversations} multi-step conversations → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=40)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    output = Path(args.output) if args.output else None
    generate_dataset(args.num_conversations, output)

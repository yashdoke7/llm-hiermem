"""
Generate Needle-in-a-Haystack conversations.

Plants a specific fact/instruction at a known turn, then tests recall
at varying distances (10, 20, 30, 40+ turns later).
"""

import json
import random
import argparse
from pathlib import Path

NEEDLES = [
    ("The secret password is 'aurora-7742'.", "What is the secret password?", "aurora-7742"),
    ("My favorite color is cerulean blue.", "What is my favorite color?", "cerulean"),
    ("The project deadline is March 15, 2025.", "When is the project deadline?", "March 15"),
    ("Use port 8443 for the staging server.", "What port should the staging server use?", "8443"),
    ("The API rate limit is 100 requests per minute.", "What is the API rate limit?", "100 requests"),
    ("My team uses the branch naming convention: feature/TICKET-description.", "What branch naming convention do we use?", "feature/TICKET"),
    ("The database backup runs at 3:00 AM UTC daily.", "When does the database backup run?", "3:00 AM UTC"),
    ("The maximum file upload size is 25MB.", "What is the maximum file upload size?", "25MB"),
    ("Our CI/CD pipeline uses GitHub Actions with self-hosted runners.", "What CI/CD tool do we use?", "GitHub Actions"),
    ("The admin email is ops-team@company.internal.", "What is the admin email?", "ops-team@company"),
]

FILLER_EXCHANGES = [
    ("How do I optimize a SQL query?", "Consider using indexes, EXPLAIN ANALYZE..."),
    ("What's the best way to handle errors in Go?", "Go uses explicit error returns..."),
    ("Explain Docker networking", "Docker has bridge, host, overlay networks..."),
    ("How do I set up CI/CD?", "Start with a pipeline config file..."),
    ("What is Kubernetes?", "Kubernetes is a container orchestration platform..."),
    ("How do I write unit tests?", "Use your language's test framework..."),
    ("Explain microservices", "Microservices break down applications..."),
    ("What is GraphQL?", "GraphQL is a query language for APIs..."),
    ("How do I handle authentication?", "Use JWT, OAuth2, or session-based auth..."),
    ("Explain event-driven architecture", "Events trigger asynchronous processing..."),
]


def generate_niah_conversation(needle_turn: int, recall_distances: list,
                                total_turns: int = 50, seed: int = None) -> dict:
    """Generate a NIAH conversation."""
    if seed is not None:
        random.seed(seed)
    
    needle_text, recall_question, expected_answer = random.choice(NEEDLES)
    
    turns = []
    checkpoints = []
    fillers = list(FILLER_EXCHANGES) * 10
    random.shuffle(fillers)
    filler_idx = 0
    
    for t in range(1, total_turns + 1):
        if t == needle_turn:
            turns.append({"turn": t, "role": "user",
                         "text": f"Important note: {needle_text}", "type": "needle"})
            turns.append({"turn": t, "role": "assistant",
                         "text": "Got it, I've noted that.", "type": "ack"})
        elif t in [needle_turn + d for d in recall_distances if needle_turn + d <= total_turns]:
            turns.append({"turn": t, "role": "user",
                         "text": recall_question, "type": "recall"})
            turns.append({"turn": t, "role": "assistant",
                         "text": "[RESPONSE_PLACEHOLDER]", "type": "recall_response"})
            checkpoints.append({
                "turn": t,
                "distance_from_needle": t - needle_turn,
                "test": f"Does response contain '{expected_answer}'?",
                "answer": True,
            })
        else:
            if filler_idx < len(fillers):
                u, a = fillers[filler_idx]
                filler_idx += 1
            else:
                u, a = f"Question {t}", f"Answer {t}"
            turns.append({"turn": t, "role": "user", "text": u, "type": "filler"})
            turns.append({"turn": t, "role": "assistant", "text": a, "type": "filler"})
    
    return {
        "conversation_id": f"niah_{seed or random.randint(0, 99999):05d}",
        "needle_turn": needle_turn,
        "needle_text": needle_text,
        "recall_question": recall_question,
        "expected_answer": expected_answer,
        "num_turns": total_turns,
        "checkpoints": checkpoints,
        "turns": turns,
    }


def generate_dataset(num_conversations: int = 30, output_path: Path = None):
    """Generate the full NIAH dataset with varying needle positions and distances."""
    output_path = output_path or Path(__file__).parent.parent / "datasets" / "needle_in_haystack"
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    for i in range(num_conversations):
        needle_pos = random.randint(2, 10)
        total = random.choice([30, 40, 50])
        distances = [5, 10, 15, 20, 25, 30, 35]
        
        conv = generate_niah_conversation(
            needle_turn=needle_pos,
            recall_distances=distances,
            total_turns=total,
            seed=i
        )
        dataset.append(conv)
    
    out_file = output_path / "conversations.json"
    out_file.write_text(json.dumps(dataset, indent=2))
    print(f"Generated {num_conversations} NIAH conversations → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    output = Path(args.output) if args.output else None
    generate_dataset(args.num_conversations, output)

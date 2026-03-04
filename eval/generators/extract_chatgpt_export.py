"""
Extract multi-turn conversations from a ChatGPT export (conversations.json).

The export uses a tree structure (parent→children mapping), not a flat list.
This script walks the tree to reconstruct linear conversation threads,
filters for long conversations, and outputs them in our benchmark format.

Usage:
    python -m eval.generators.extract_chatgpt_export \
        --input eval/datasets/constraint_tracking/real_exports/chatgpt/conversations.json \
        --output eval/datasets/constraint_tracking/real_exports/chatgpt_extracted.json \
        --min-turns 10
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_conversations(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def walk_tree(mapping: dict) -> list[dict]:
    """Walk the tree from root to leaf (following current_node path) to get linear conversation."""
    # Find the root node (no parent or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if not root_id:
        return []

    # Walk from root following children (take first child at each branch)
    messages = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if not node:
            break

        msg = node.get("message")
        if msg and msg.get("content"):
            author = msg.get("author", {})
            role = author.get("role", "")
            content = msg["content"]

            # Extract text from parts
            parts = content.get("parts", [])
            text_parts = []
            for part in parts:
                if isinstance(part, str) and part.strip():
                    text_parts.append(part.strip())
                elif isinstance(part, dict):
                    # Could be image/file reference — skip
                    pass

            text = "\n".join(text_parts)

            if text and role in ("user", "assistant"):
                entry = {
                    "role": role,
                    "content": text,
                }
                # Add model info for assistant messages
                if role == "assistant":
                    model = msg.get("metadata", {}).get("model_slug", "")
                    if model:
                        entry["model"] = model
                messages.append(entry)

        # Move to first child
        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages


def extract_conversations(
    data: list, min_turns: int = 10, max_turns: int = 0
) -> list[dict]:
    """Extract and filter conversations from ChatGPT export."""
    extracted = []

    for convo in data:
        title = convo.get("title", "Untitled")
        mapping = convo.get("mapping", {})
        create_time = convo.get("create_time", 0)

        messages = walk_tree(mapping)

        # Count user turns
        user_turns = sum(1 for m in messages if m["role"] == "user")

        if user_turns < min_turns:
            continue

        # Trim if max_turns specified
        if max_turns > 0:
            trimmed = []
            user_count = 0
            for m in messages:
                if m["role"] == "user":
                    user_count += 1
                if user_count > max_turns:
                    break
                trimmed.append(m)
            messages = trimmed

        # Collect models used
        models = list(
            set(m.get("model", "") for m in messages if m.get("model"))
        )

        extracted.append(
            {
                "id": convo.get("id", convo.get("conversation_id", "")),
                "title": title,
                "source": "chatgpt_export",
                "created": datetime.fromtimestamp(create_time).isoformat()
                if create_time
                else None,
                "models_used": models,
                "user_turns": sum(1 for m in messages if m["role"] == "user"),
                "total_messages": len(messages),
                "messages": messages,
            }
        )

    # Sort by turn count descending
    extracted.sort(key=lambda c: c["user_turns"], reverse=True)
    return extracted


def print_summary(conversations: list):
    print(f"\nExtracted {len(conversations)} conversations:")
    for i, c in enumerate(conversations):
        print(
            f"  [{i}] \"{c['title']}\" — {c['user_turns']} user turns, "
            f"{c['total_messages']} total msgs, models: {c['models_used']}"
        )

    if conversations:
        turns = [c["user_turns"] for c in conversations]
        print(
            f"\n  Turn stats: min={min(turns)}, max={max(turns)}, "
            f"avg={sum(turns)/len(turns):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract conversations from ChatGPT export"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to conversations.json from ChatGPT export",
    )
    parser.add_argument(
        "--output", required=True, help="Output JSON file path"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=10,
        help="Minimum user turns to include (default: 10)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Max user turns to keep per conversation (0 = no limit)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't write output",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    data = load_conversations(args.input)
    print(f"Found {len(data)} total conversations in export")

    extracted = extract_conversations(data, args.min_turns, args.max_turns)
    print_summary(extracted)

    if not args.summary_only and extracted:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(extracted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved to {output_path}")
    elif not extracted:
        print("\nNo conversations matched the criteria.")


if __name__ == "__main__":
    main()

"""
Script to generate UI cross-validation prompts for checking judge quality.

This script extracts a random sample of evaluated responses from a benchmark run
and formats them into Ready-to-Paste prompts for ChatGPT or Gemini UI.
"""

import json
import random
import argparse
from pathlib import Path

def generate_ui_prompts(results_dir: Path, num_samples: int = 15, output_file: Path = None):
    results_path = results_dir / "results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        return

    data = json.loads(results_path.read_text(encoding="utf-8"))
    
    # Collect all checkpoints
    samples = []
    for system_name, convos in data.items():
        for conv in convos:
            turn_logs = conv.get("turn_logs", [])
            checkpoints = conv.get("checkpoints", [])
            
            for cp in checkpoints:
                turn_num = cp.get("turn", 0)
                response = None
                user_msg = ""
                
                for t in turn_logs:
                    if t.get("turn") == turn_num:
                        response = t.get("response", "")
                        break
                
                if not response or response.startswith("ERROR:"):
                    continue
                
                for t in turn_logs:
                    if t.get("turn") == turn_num - 1 or t.get("turn") == turn_num:
                        if t.get("user"):
                            user_msg = t["user"]
                            break
                            
                samples.append({
                    "system": system_name,
                    "turn": turn_num,
                    "response": response,
                    "constraint": cp.get("constraint_tested", ""),
                    "test": cp.get("test", ""),
                    "user_msg": user_msg
                })

    if not samples:
        print("No valid checkpoints found.")
        return

    # Randomly select
    random.seed(42) # For reproducibility
    chosen = random.sample(samples, min(num_samples, len(samples)))
    
    # Generate Prompt Markdown
    out = [
        "# UI Cross-Validation Prompts\n",
        "Copy and paste the text blocks below into ChatGPT or Gemini to manually cross-validate the judge's scoring.\n"
        "Keep track of the scores they give you and compare them to your automated run!\n\n"
        "---\n"
    ]
    
    for i, s in enumerate(chosen):
        user_context = f"\nUSER'S QUESTION IN THIS TURN:\n{s['user_msg'][:500]}\n" if s['user_msg'] else ""
        test_instruction = (f"\nSpecifically evaluate: {s['test']}" if s['test'] else "")
        
        prompt = (
            f"You are evaluating how well an AI assistant follows user-specified rules.\n\n"
            f"RULES the user asked the assistant to follow:\n{s['constraint']}\n"
            f"{user_context}"
            f"\nASSISTANT'S RESPONSE:\n{s['response'][:4000]}\n\n"
            f"Rate the response on a scale of 1 to 10 for how well it follows the rules:{test_instruction}\n"
            f"  10 = Perfectly follows every rule stated\n"
            f"   8 = Follows most rules, one minor lapse\n"
            f"   6 = Follows some rules but misses others\n"
            f"   4 = Mostly ignores the rules with occasional correct elements\n"
            f"   2 = Almost completely ignores the rules\n"
            f"   1 = Completely ignores all rules\n\n"
            f"Notes:\n"
            f"- If the question genuinely does not call for code and none is produced, score 8 (constraint acknowledged but not directly applicable)\n"
            f"- If the question calls for code or code is produced, the constraint rules MUST be visible in the code to score above 5\n"
            f"- If code uses the WRONG pattern (e.g. pandas instead of polars), score 2-3 regardless of other quality\n"
            f"- Be consistent: judge the pattern of constraint adherence across the whole response\n\n"
            f"Output your score as: SCORE: <number>\n"
            f"Then optionally one sentence of reasoning."
        )
        
        out.append(f"## Sample {i+1} ({s['system']}, turn {s['turn']})\n")
        out.append("```text\n")
        out.append(prompt)
        out.append("\n```\n\n---\n")

    if output_file is None:
        output_file = results_dir / "ui_cross_validation_prompts.md"
        
    output_file.write_text("".join(out), encoding="utf-8")
    print(f"Generated {len(chosen)} prompts for UI cross-validation at: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UI Cross Validation Prompts")
    parser.add_argument("results_dir", type=str, help="Path to raw benchmark results directory (e.g., results/raw/benchmarks/qwen14b_run_v4)")
    parser.add_argument("--samples", type=int, default=15, help="Number of random samples to generate")
    args = parser.parse_args()
    
    generate_ui_prompts(Path(args.results_dir), args.samples)

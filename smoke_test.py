"""
Quick smoke test — run 2 conversations × 10 turns each on HierMem + raw_llm baseline.
Verifies the benchmark pipeline works before full evaluation.
"""
import json
import time
from pathlib import Path

from core.pipeline import HierMemPipeline
from baselines.raw_llm import RawLLMBaseline

# Load first 2 constraint conversations
dataset_path = Path("eval/datasets/constraint_tracking/conversations.json")
all_convos = json.loads(dataset_path.read_text())
test_convos = all_convos[:2]

# Trim to first 10 turns each
for conv in test_convos:
    conv["turns"] = [t for t in conv["turns"] if t.get("turn", 0) <= 10]
    conv["checkpoints"] = [c for c in conv["checkpoints"] if c.get("turn", 0) <= 10]

systems = {
    "hiermem": lambda: HierMemPipeline.create(),
    "raw_llm": lambda: RawLLMBaseline(),
}

results = {}

for name, factory in systems.items():
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    
    system_results = []
    
    for conv in test_convos:
        system = factory()
        conv_id = conv["conversation_id"]
        print(f"  Conversation: {conv_id}")
        
        turn_results = []
        start = time.time()
        
        for turn_data in conv["turns"]:
            if turn_data.get("role") == "user":
                user_msg = turn_data["text"]
                turn_num = turn_data.get("turn", 0)
                
                try:
                    result = system.process_turn(user_msg)
                    response = result.assistant_response if hasattr(result, 'assistant_response') else str(result)
                    print(f"    Turn {turn_num}: OK ({len(response)} chars)")
                except Exception as e:
                    response = f"ERROR: {e}"
                    print(f"    Turn {turn_num}: FAILED - {e}")
                
                turn_results.append({
                    "turn": turn_num,
                    "user": user_msg[:80],
                    "response": response[:200],
                })
        
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")
        
        # Check constraints if HierMem
        if name == "hiermem" and hasattr(system, 'constraint_store'):
            constraints = system.constraint_store.get_display_text()
            print(f"  Constraints: {constraints[:200]}")
        
        system_results.append({
            "conversation_id": conv_id,
            "turns": turn_results,
            "elapsed_seconds": elapsed,
        })
    
    results[name] = system_results

# Save
output_path = Path("results/smoke_test.json")
output_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {output_path}")
print("\nSmoke test complete!")

"""
Quick smoke test — run 2 conversations × 10 turns each on HierMem + raw_llm baseline.
Verifies the benchmark pipeline works before full evaluation.
"""
import json
import time
from pathlib import Path
from datetime import datetime

from core.pipeline import HierMemPipeline
from baselines.raw_llm import RawLLMBaseline
import config

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

# Build descriptive output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
provider = config.DEFAULT_PROVIDER
main_model = config.MAIN_LLM_MODEL.split("/")[-1].replace(":", "-")
output_dir = Path("results/raw/smoke_tests")
output_dir.mkdir(parents=True, exist_ok=True)

run_meta = {
    "test_type": "smoke_test",
    "timestamp": timestamp,
    "provider": provider,
    "models": {
        "main": config.MAIN_LLM_MODEL,
        "curator": config.CURATOR_MODEL,
        "summarizer": config.SUMMARIZER_MODEL,
    },
    "config": {
        "context_budget": config.TOTAL_CONTEXT_BUDGET,
        "segment_size": config.SEGMENT_SIZE,
        "max_constraints": config.MAX_CONSTRAINTS,
    },
    "conversations": len(test_convos),
    "max_turns": 10,
}

print(f"Provider: {provider} | Main: {config.MAIN_LLM_MODEL} | Curator: {config.CURATOR_MODEL}")

results = {"meta": run_meta}

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
        daily_limit_hit = False
        
        for turn_data in conv["turns"]:
            if turn_data.get("role") == "user":
                user_msg = turn_data["text"]
                turn_num = turn_data.get("turn", 0)
                
                try:
                    result = system.process_turn(user_msg)
                    response = result.assistant_response if hasattr(result, 'assistant_response') else str(result)
                    print(f"    Turn {turn_num}: OK ({len(response)} chars)")
                except Exception as e:
                    err_msg = str(e).lower()
                    if "daily" in err_msg or "quota exhausted" in err_msg:
                        print(f"    Turn {turn_num}: DAILY LIMIT HIT — aborting")
                        print("    ⚠ Groq daily quota exhausted. Switch provider or wait for reset.")
                        daily_limit_hit = True
                        break
                    response = f"ERROR: {e}"
                    print(f"    Turn {turn_num}: FAILED - {e}")
                
                turn_results.append({
                    "turn": turn_num,
                    "user": user_msg[:80],
                    "response": response[:200],
                })
        
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")
        
        conv_result = {
            "conversation_id": conv_id,
            "turns": turn_results,
            "elapsed_seconds": round(elapsed, 1),
        }
        
        # Capture system state
        if name == "hiermem" and hasattr(system, 'constraint_store'):
            constraints = system.constraint_store.get_display_text()
            print(f"  Constraints: {constraints[:200]}")
            conv_result["constraints"] = constraints
            conv_result["segments"] = len(system.archive.l0_directory)
        
        system_results.append(conv_result)
        
        if daily_limit_hit:
            print("  ⚠ Skipping remaining conversations for this system")
            break
    
    results[name] = system_results

# Save with descriptive filename
filename = f"smoke_{provider}_{main_model}_{timestamp}.json"
output_path = output_dir / filename
output_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {output_path}")
print("Smoke test complete!")

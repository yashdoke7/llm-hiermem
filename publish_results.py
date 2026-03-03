"""
Publish best results — Extract key metrics from raw results and save to published/.

Usage:
    python publish_results.py results/raw/smoke_tests/smoke_ollama_llama3.1-8b_20260303_*.json
    python publish_results.py results/raw/benchmarks/constraint_tracking_groq_*/results.json
"""
import json
import sys
from pathlib import Path
from datetime import datetime


def extract_summary(raw_path: Path) -> dict:
    """Extract key metrics from a raw result file."""
    data = json.loads(raw_path.read_text())
    meta = data.get("meta", {})
    
    provider = meta.get("provider", "unknown")
    models = meta.get("models", {})
    main_model = models.get("main", "unknown")
    
    summary = {
        "model": main_model,
        "provider": provider,
        "timestamp": meta.get("timestamp", ""),
        "source_file": str(raw_path),
        "systems": {},
    }
    
    for system_name in ["hiermem", "raw_llm", "rag", "rag_summary", "memgpt_style"]:
        if system_name not in data:
            continue
        
        sys_data = data[system_name]
        if not isinstance(sys_data, list):
            continue
        
        total_turns = 0
        total_ok = 0
        total_time = 0
        all_constraints = []
        
        for conv in sys_data:
            turns = conv.get("turns", [])
            total_turns += len(turns)
            total_ok += sum(1 for t in turns if not t.get("response", "").startswith("ERROR"))
            total_time += conv.get("elapsed_seconds", 0)
            if conv.get("constraints"):
                all_constraints.append(conv["constraints"])
        
        summary["systems"][system_name] = {
            "conversations": len(sys_data),
            "total_turns": total_turns,
            "successful_turns": total_ok,
            "success_rate": round(total_ok / max(total_turns, 1) * 100, 1),
            "total_time_seconds": round(total_time, 1),
            "avg_turn_latency": round(total_time / max(total_ok, 1), 1),
            "constraints_captured": all_constraints,
        }
    
    return summary


def publish(raw_path: Path):
    """Publish a summary to results/published/models/."""
    summary = extract_summary(raw_path)
    
    model_slug = summary["model"].split("/")[-1].replace(":", "-")
    provider = summary["provider"]
    
    pub_dir = Path("results/published/models")
    pub_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if existing published result for this model is worse
    existing = list(pub_dir.glob(f"{provider}_{model_slug}_*.json"))
    
    # Get hiermem success rate from new summary
    new_rate = summary.get("systems", {}).get("hiermem", {}).get("success_rate", 0)
    
    should_publish = True
    for ef in existing:
        old = json.loads(ef.read_text())
        old_rate = old.get("systems", {}).get("hiermem", {}).get("success_rate", 0)
        if old_rate >= new_rate:
            print(f"  Existing result ({old_rate}%) >= new ({new_rate}%), skipping")
            should_publish = False
            break
    
    if should_publish:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = pub_dir / f"{provider}_{model_slug}_{ts}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Published: {out_path}")
        print(f"  HierMem success: {new_rate}%")
        for name, sys_info in summary["systems"].items():
            print(f"  {name}: {sys_info['success_rate']}% "
                  f"({sys_info['successful_turns']}/{sys_info['total_turns']} turns, "
                  f"{sys_info['avg_turn_latency']}s/turn)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python publish_results.py <raw_result.json> [...]")
        sys.exit(1)
    
    for path_str in sys.argv[1:]:
        for p in Path(".").glob(path_str) if "*" in path_str else [Path(path_str)]:
            if p.exists():
                print(f"\nProcessing: {p}")
                publish(p)
            else:
                print(f"Not found: {p}")

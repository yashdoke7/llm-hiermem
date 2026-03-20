import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    codex_file = "results/raw/benchmarks/qwen14b_run_v4/metrics_codex.json"
    out_png = "results/raw/benchmarks/qwen14b_run_v4/codex_degradation_chart.png"
    
    if not os.path.exists(codex_file):
        print(f"File not found: {codex_file}")
        return

    with open(codex_file, "r") as f:
        data = json.load(f).get("dataset_evaluation", {})

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    colors = {
        "hier_mem": "#1f77b4",  # Blue
        "raw_llm": "#ff7f0e",   # Orange
        "rag": "#d62728",       # Red
        "rag_summary": "#2ca02c"# Green
    }
    
    labels = {
        "hier_mem": "HierMem (Ours)",
        "raw_llm": "Raw LLM (Full History)",
        "rag": "RAG (Top-20)",
        "rag_summary": "RAG + Summary"
    }

    for system_key, details in data.items():
        if system_key in ["final_rankings", "head_to_head_summary"]:
            continue
            
        checkpoints = details.get("checkpoints", [])
        if not checkpoints:
            continue
            
        # Extract X and Y
        turns = [c["turn"] for c in checkpoints]
        scores = [c["score"] for c in checkpoints]
        
        plt.plot(turns, scores, marker='o', linewidth=2.5, 
                 label=labels.get(system_key, system_key),
                 color=colors.get(system_key, "black"))

    plt.title("Constraint Degradation Over 50 Turns (Macro-Eval by GPT-Codex)", fontsize=14, pad=15)
    plt.xlabel("Conversation Turn Number", fontsize=12)
    plt.ylabel("Constraint Preservation Score (1-10)", fontsize=12)
    plt.ylim(0, 10.5)
    plt.xticks(range(0, 101, 10))
    plt.legend(title="Algorithm Architecture")
    plt.tight_layout()
    
    plt.savefig(out_png, dpi=300)
    print(f"Saved holistic visualization to {out_png}")

if __name__ == "__main__":
    main()

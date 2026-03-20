import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Pricing configuration (Cost per 1 Million Tokens)
# These represent typical market rates for large highly-capable models (Main) vs small fast models (Curator)
MAIN_MODEL_PRICE_PER_1M = 0.80  # e.g., representing 14B/32B parameters context window price
CURATOR_MODEL_PRICE_PER_1M = 0.10 # e.g., representing 3B parameters context window price

# Estimated Context size evaluated by the Curator on summarization triggers
CURATOR_TOKENS_PER_SUMMARY = 2500 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()

    results_dir = args.results
    files = glob.glob(os.path.join(results_dir, "*_detailed.json"))
    
    if not files:
        print(f"No *_detailed.json files found in {results_dir}")
        return

    data_records = []

    for file in files:
        system_name = os.path.basename(file).split("_detailed")[0]
        with open(file, 'r', encoding='utf-8') as f:
            runs = json.load(f)
        
        for run in runs:
            cumulative_main_cost = 0.0
            cumulative_curator_cost = 0.0
            
            for i, turn_data in enumerate(run.get("turn_logs", [])):
                turn_num = turn_data["turn"]
                tokens = turn_data.get("pipeline_details", {}).get("context_tokens_used", turn_data.get("context_tokens", 0))                
                # Main Model Cost
                cost_this_turn = (tokens / 1_000_000) * MAIN_MODEL_PRICE_PER_1M
                cumulative_main_cost += cost_this_turn
                
                # Curator/Summarizer auxiliary cost
                # - HierMem: approximate curator+summarizer calls every 5 turns
                # - rag_summary: summary update happens every turn (max_tokens ~300)
                if system_name == "hiermem" and (i + 1) % 5 == 0:
                    curator_cost_this_turn = (CURATOR_TOKENS_PER_SUMMARY / 1_000_000) * CURATOR_MODEL_PRICE_PER_1M
                    cumulative_curator_cost += curator_cost_this_turn
                elif system_name == "rag_summary":
                    rag_summary_aux_tokens = 300
                    curator_cost_this_turn = (rag_summary_aux_tokens / 1_000_000) * CURATOR_MODEL_PRICE_PER_1M
                    cumulative_curator_cost += curator_cost_this_turn
                    
                cumulative_cost = cumulative_main_cost + cumulative_curator_cost
                
                data_records.append({
                    "System": system_name,
                    "Turn": turn_num,
                    "Context Tokens": tokens,
                    "Main Cost ($)": cumulative_main_cost,
                    "Curator Cost ($)": cumulative_curator_cost,
                    "Cumulative Total Cost ($)": cumulative_cost
                })

    df = pd.DataFrame(data_records)
    
    # Average across all conversations for a given system and turn
    # Need to filter for numeric columns only to avoid future warning
    avg_df = df.groupby(["System", "Turn"])[["Context Tokens", "Main Cost ($)", "Curator Cost ($)", "Cumulative Total Cost ($)"]].mean().reset_index()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # 1. Line plot: Cumulative Cost vs Turn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_df, x="Turn", y="Cumulative Total Cost ($)", hue="System", marker="o", linewidth=2)
    plt.title("Cumulative API Context Cost vs Turn (Main + Curator LLMs)", fontweight="bold")
    plt.ylabel("Cumulative Cost ($)")
    plt.xlabel("Conversation Turn")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "api_cost_vs_turn.png"), dpi=300)
    plt.close()
    
    # 2. Bar plot: Total Costs separated by Main and Curator
    max_turn_df = avg_df.loc[avg_df.groupby("System")["Turn"].idxmax()]
    
    plt.figure(figsize=(8, 6))
    systems = max_turn_df["System"].values
    main_costs = max_turn_df["Main Cost ($)"].values
    curator_costs = max_turn_df["Curator Cost ($)"].values
    
    x = np.arange(len(systems))
    width = 0.5
    
    plt.bar(x, main_costs, width, label='Main LLM Cost', color='skyblue', edgecolor='black')
    plt.bar(x, curator_costs, width, bottom=main_costs, label='Curator LLM Cost (HierMem Only)', color='salmon', edgecolor='black')
    
    plt.xlabel('System')
    plt.ylabel('Total API Cost ($) per 50-turn Conversation')
    plt.title('Total API Cost Breakdown', fontweight="bold")
    plt.xticks(x, systems)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "api_cost_breakdown.png"), dpi=300)
    plt.close()

    print(f"\nCost Analysis visualizer completed successfully! \nSaved PNG files to: {results_dir}\n")
    
    print("--- Estimated API Cost per 50-Turn Conversation ---")
    for _, row in max_turn_df.iterrows():
        print(f"[{row['System'].upper()}] Total: ${row['Cumulative Total Cost ($)']:.4f} "
              f"(Main LLM: ${row['Main Cost ($)']:.4f}, Curator LLM: ${row['Curator Cost ($)']:.4f})")

if __name__ == "__main__":
    main()

import os
import json
import glob
import shutil
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Please install required packages: pip install pandas seaborn matplotlib")
    exit(1)

# Style configuration for Research Grade Paper
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

SYSTEM_LABELS = {
    "hiermem": "HierMem",
    "raw_llm": "Raw LLM",
    "rag": "RAG",
    "rag_summary": "RAG Summary",
}
SYSTEM_COLORS = {
    "HierMem": "#1f9d55",       # Deep Green
    "Raw LLM": "#e4572e",       # Red-Orange
    "RAG": "#3a86ff",           # Blue
    "RAG Summary": "#ff9f1c",   # Amber
}

# --- Pricing Configuration ---
PRICE_MAIN = 0.40    # 14B Model ($/1M)
PRICE_CURATOR = 0.10 # 3B Model ($/1M)
PRICE_EMBED = 0.05   # Embedding ($/1M)
CONTEXT_BUDGET = 32768  # 32k KV cache ceiling

def load_all_data(results_root: Path):
    dataset_dirs = [d for d in results_root.glob("dataset_*") if d.is_dir()]
    print(f"Analyzing {len(dataset_dirs)} datasets in {results_root}")

    turn_records = []
    run_records = []

    for d in dataset_dirs:
        results_file = d / "results.json"
        laaj_file = d / "laaj.json"
        
        if not results_file.exists() or not laaj_file.exists():
            continue
            
        with open(results_file, 'r', encoding='utf-8') as f:
            runs = json.load(f)
        with open(laaj_file, 'r', encoding='utf-8') as f:
            laaj_data = json.load(f)

        identity_map = laaj_data.get("system_identity_reveal", {})
        reverse_map = {alias: real_name for alias, real_name in identity_map.items()}
        
        judge_scores_map = {}
        for ranking in laaj_data.get("overall_ranking", []):
            alias = ranking["system"]
            real_name = reverse_map.get(alias, alias)
            judge_scores_map[real_name] = ranking.get("average_weighted_score", 0.0)
            
        eval_dict = laaj_data.get("system_evaluations", {})

        for system_name, run_list in runs.items():
            if system_name not in SYSTEM_LABELS:
                continue
                
            human_label = SYSTEM_LABELS[system_name]

            for run in run_list:
                cumulative_cost = 0.0
                total_main_cost = 0.0
                total_curator_cost = 0.0
                total_embed_cost = 0.0
                
                total_main_tokens = 0
                total_curator_tokens = 0
                
                latencies = []

                for i, tl in enumerate(run.get("turn_logs", run.get("results", []))):
                    turn_idx = tl.get("turn", 0)
                    pd_details = tl.get("pipeline_details", {})
                    lat = tl.get("latency_seconds", 0.0)
                    if lat > 0:
                        latencies.append(lat)
                    
                    # 1. Main Cost
                    ctx = tl.get("context_tokens", 0)
                    actual_ctx = pd_details.get("context_tokens_used", 0) or ctx
                    main_tokens = actual_ctx + max(1, len(tl.get("response", "")) // 4)
                    main_cost = (main_tokens / 1_000_000) * PRICE_MAIN
                    total_main_tokens += main_tokens
                    
                    # 2. Curator Cost (Telemetry-based)
                    curator_tokens = 0
                    if system_name == "hiermem":
                        curator_tokens += pd_details.get("curator_total_tokens", 0)
                        curator_tokens += pd_details.get("postproc_summary_tokens", 0)
                        curator_tokens += pd_details.get("postproc_extract_tokens", 0)
                        curator_tokens += pd_details.get("violation_retry_tokens", 0)
                    elif system_name == "rag_summary":
                        curator_tokens += 300 # summary updates
                    curator_cost = (curator_tokens / 1_000_000) * PRICE_CURATOR
                    total_curator_tokens += curator_tokens
                    
                    # 3. Embedding Cost
                    embed_calls = len(pd_details.get("semantic_queries", []))
                    embed_tokens = embed_calls * 128 # estimate
                    embed_cost = (embed_tokens / 1_000_000) * PRICE_EMBED

                    turn_total = main_cost + curator_cost + embed_cost
                    cumulative_cost += turn_total
                    total_main_cost += main_cost
                    total_curator_cost += curator_cost
                    total_embed_cost += embed_cost

                    turn_records.append({
                        "Dataset": d.name,
                        "System": human_label,
                        "Turn": turn_idx,
                        "Context Tokens": actual_ctx,
                        "Latency (s)": lat,
                        "Cumulative Cost ($)": cumulative_cost
                    })

                # --- Accumulated context: total user+assistant text generated so far ---
                accumulated = 0
                for j, tl2 in enumerate(run.get("turn_logs", run.get("results", []))):
                    user_text = tl2.get("user", "")
                    resp_text = tl2.get("response", "")
                    accumulated += max(1, len(user_text) // 4) + max(1, len(resp_text) // 4)
                    turn_records.append({
                        "Dataset": d.name,
                        "System": "Accumulated (All)",
                        "Turn": tl2.get("turn", 0),
                        "Accumulated Tokens": accumulated,
                    })

                # Map Judge scores to correct turns
                system_alias = next((a for a, r in reverse_map.items() if r == system_name), "")
                checkpoints = eval_dict.get(system_alias, {}).get("checkpoints", [])
                for cp in checkpoints:
                    turn_records.append({
                        "Dataset": d.name,
                        "System": human_label,
                        "Checkpoint Turn": cp["checkpoint_turn"],
                        "Judge Score": cp["weighted_score"]
                    })

                # Compute compression ratio and token savings
                avg_ctx_sent = total_main_tokens / max(len(run.get("turn_logs", [1])), 1)
                compression_ratio = accumulated / max(avg_ctx_sent, 1) if accumulated > 0 else 1.0
                savings_vs_raw = 0.0  # computed later in aggregate

                run_records.append({
                    "Dataset": d.name,
                    "System": human_label,
                    "Total Cost ($)": cumulative_cost,
                    "Main Cost ($)": total_main_cost,
                    "Curator Cost ($)": total_curator_cost,
                    "Embed Cost ($)": total_embed_cost,
                    "14B Tokens": total_main_tokens,
                    "3B Tokens": total_curator_tokens,
                    "Avg Latency (s)": np.mean(latencies) if latencies else 0.0,
                    "Final Judge Score": judge_scores_map.get(system_name, 0.0),
                    "Accumulated Tokens": accumulated,
                    "Compression Ratio": round(compression_ratio, 2),
                })

    return pd.DataFrame(turn_records), pd.DataFrame(run_records)

def generate_visuals(df_turn, df_run, out_dir, prefix="Overall"):
    """Core plotting logic shared between individual and whole modes."""
    # 1. Quality Ribbon
    df_scores = df_turn.dropna(subset=["Judge Score"])
    if not df_scores.empty:
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=df_scores, x="Checkpoint Turn", y="Judge Score", hue="System", 
                     palette=SYSTEM_COLORS, marker="o", linewidth=2.5, errorbar=("ci", 95) if prefix=="Overall" else None)
        plt.title(f"{prefix}: Constraint Compliance vs Turns", fontweight="bold")
        plt.ylim(-0.5, 10.5); plt.ylabel("Score (1-10)")
        plt.legend(title="", title_fontsize=12, fontsize=11); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_1_Quality.png", dpi=300)
        plt.close()

    # 2. Token Complexity
    df_ctx = df_turn.dropna(subset=["Context Tokens"])
    if not df_ctx.empty:
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=df_ctx, x="Turn", y="Context Tokens", hue="System", 
                     palette=SYSTEM_COLORS, linewidth=2.5, errorbar=("ci", 95) if prefix=="Overall" else None)
        plt.title(f"{prefix}: Memory Complexity (Context Payload size)", fontweight="bold")
        plt.legend(title="", fontsize=11); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_2_Context.png", dpi=300)
        plt.close()

    # 3. Pareto Efficiency (Offset adjusted)
    if not df_run.empty:
        plt.figure(figsize=(10, 6))
        agg_run = df_run.groupby("System").mean(numeric_only=True).reset_index()
        sns.scatterplot(data=agg_run, x="Total Cost ($)", y="Final Judge Score", 
                        hue="System", palette=SYSTEM_COLORS, s=300, edgecolor="black")
        
        for i, row in agg_run.iterrows():
            offset_y = 0.2 if 'HierMem' in row["System"] else -0.3
            plt.text(row["Total Cost ($)"], row["Final Judge Score"] + offset_y, 
                     row["System"], fontsize=11, fontweight="semibold", ha='center')
        
        plt.title(f"{prefix}: Cost-Quality Pareto Front", fontweight="bold")
        plt.ylabel("Average LAAJ Score (1-10)"); plt.xlabel("Total API Cost per Session ($)")
        plt.grid(True, alpha=0.3); plt.legend([],[], frameon=False)
        plt.ylim(max(0, agg_run["Final Judge Score"].min() - 1), 10.5)
        plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_3_Pareto.png", dpi=300)
        plt.close()

        # 4. Cost Breakdown Stacked Bar
        if prefix == "Overall":
            melted = agg_run.melt(id_vars=["System"], value_vars=["Main Cost ($)", "Curator Cost ($)", "Embed Cost ($)"], 
                                  var_name="Cost Type", value_name="Avg Cost")
            plt.figure(figsize=(9, 6))
            sns.barplot(data=melted, x="System", y="Avg Cost", hue="Cost Type", palette="viridis")
            plt.title("API Cost Distribution (14B vs 3B vs Embeddings)", fontweight="bold")
            plt.ylabel("Avg Cost per Session ($)"); plt.xlabel("")
            plt.xticks(rotation=15)
            plt.tight_layout(); plt.savefig(out_dir / "Overall_4_Cost_Breakdown.png", dpi=300)
            plt.close()
            
    # 5. Latency Line Plot (Latency vs Turn)
    df_lat = df_turn[df_turn["Latency (s)"] > 0]
    if not df_lat.empty:
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=df_lat, x="Turn", y="Latency (s)", hue="System", 
                     palette=SYSTEM_COLORS, linewidth=2.0, errorbar=("ci", 95) if prefix=="Overall" else None)
        plt.title(f"{prefix}: Response Latency over Time", fontweight="bold")
        plt.legend(title="", fontsize=11); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_5_Latency_Trend.png", dpi=300)
        plt.close()

    # 6. Latency BoxPlot (Overall Distribution)
    if not df_lat.empty and prefix == "Overall":
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df_lat, x="System", y="Latency (s)", palette=SYSTEM_COLORS, showfliers=False)
        plt.title("Distribution of Response Latencies", fontweight="bold")
        plt.ylabel("Latency per turn (s)"); plt.xlabel("")
        plt.tight_layout(); plt.savefig(out_dir / "Overall_6_Latency_Boxplot.png", dpi=300)
        plt.close()

    # 7. Context Pressure: Accumulated vs Sent vs 32k Ceiling
    df_ctx_sent = df_turn.dropna(subset=["Context Tokens"])
    df_accum = df_turn.dropna(subset=["Accumulated Tokens"]) if "Accumulated Tokens" in df_turn.columns else pd.DataFrame()
    if not df_ctx_sent.empty:
        plt.figure(figsize=(10, 6))

        # Plot per-system context sent lines
        for sys_name in ["HierMem", "Raw LLM", "RAG", "RAG Summary"]:
            sys_data = df_ctx_sent[df_ctx_sent["System"] == sys_name]
            if sys_data.empty:
                continue
            if prefix == "Overall":
                agg = sys_data.groupby("Turn")["Context Tokens"].mean().reset_index()
            else:
                agg = sys_data.sort_values("Turn")
            plt.plot(agg["Turn"], agg["Context Tokens"], linewidth=2.2,
                     label=sys_name, color=SYSTEM_COLORS.get(sys_name, "#888888"), alpha=0.9)

        # Plot accumulated context (dashed gray)
        if not df_accum.empty:
            accum_data = df_accum.drop_duplicates(subset=["Dataset", "Turn"])
            if prefix == "Overall":
                accum_agg = accum_data.groupby("Turn")["Accumulated Tokens"].mean().reset_index()
            else:
                accum_agg = accum_data.sort_values("Turn")
            plt.plot(accum_agg["Turn"], accum_agg["Accumulated Tokens"],
                     linewidth=2.5, linestyle="--", color="#666666", alpha=0.8,
                     label="Total Conversation")

        # 32k ceiling line
        plt.axhline(y=CONTEXT_BUDGET, color="#dc2626", linestyle=":", linewidth=2.0,
                    label=f"{CONTEXT_BUDGET//1024}k Context Limit", alpha=0.7)

        plt.title(f"{prefix}: Context Pressure — Sent vs Total Generated", fontweight="bold")
        plt.xlabel("Turn Number"); plt.ylabel("Tokens")
        plt.legend(fontsize=10, loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_7_Context_Pressure.png", dpi=300)
        plt.close()

    # 8. Compression Ratio Bar Chart
    if not df_run.empty and "Compression Ratio" in df_run.columns:
        agg_run = df_run.groupby("System").mean(numeric_only=True).reset_index()
        agg_run = agg_run[agg_run["Compression Ratio"] > 0]
        if not agg_run.empty and prefix == "Overall":
            plt.figure(figsize=(9, 5))
            bars = plt.bar(agg_run["System"], agg_run["Compression Ratio"],
                          color=[SYSTEM_COLORS.get(s, "#888888") for s in agg_run["System"]],
                          alpha=0.9, edgecolor="black", linewidth=0.8)
            for bar, val in zip(bars, agg_run["Compression Ratio"]):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f"{val:.1f}x", ha="center", fontsize=12, fontweight="bold")
            plt.title("Context Compression Ratio (Higher = More Efficient)", fontweight="bold")
            plt.ylabel("Compression Ratio (Accumulated / Avg Sent)")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "Overall_8_Compression_Ratio.png", dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root benchmarks directory")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = root / "publish_graphs"
    
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    df_turn, df_run = load_all_data(root)
    if df_run.empty:
        print("No results found.")
        return

    # A. Aggregate Graphs
    print("\n--- Generating Master Figures ---")
    generate_visuals(df_turn, df_run, out_dir, "Overall")

    # B. Individual Graphs
    print("--- Generating Individual Dataset Figures ---")
    for dataset in df_run["Dataset"].unique():
        d_out = out_dir / "individual" / dataset
        d_out.mkdir(parents=True, exist_ok=True)
        generate_visuals(df_turn[df_turn["Dataset"]==dataset], df_run[df_run["Dataset"]==dataset], d_out, dataset)

    # C. Print Master Table for the UI
    print("\n" + "="*80)
    print("MASTER COMPARISON TABLE (Avg per 50-turn conversation)")
    print("="*80)
    agg_run = df_run.groupby("System").mean(numeric_only=True).reset_index()
    
    # Format terminal output
    header = f"{'System':<20} | {'14B Tokens':>12} | {'3B Tokens':>10} | {'Total Cost':>10} | {'Avg Latency':>12} | {'Compress':>10}"
    print(header)
    print("-" * 90)
    for i, row in agg_run.iterrows():
        cr = row.get('Compression Ratio', 0)
        cr_str = f"{cr:.1f}x" if cr > 0 else "N/A"
        print(f"{row['System']:<20} | {int(row['14B Tokens']):>12,} | {int(row['3B Tokens']):>10,} | ${row['Total Cost ($)']:>9.2f} | {row['Avg Latency (s)']:>10.2f}s | {cr_str:>10}")
    print("="*80)
    print(f"\n✅ All graphs saved to: {out_dir}")

if __name__ == "__main__":
    main()

# Overleaf Figure Guide

This file maps the prepared figures to paper sections.

## Prepared figure files

- [docs/figures/overall_quality.png](figures/overall_quality.png)
- [docs/figures/overall_pareto.png](figures/overall_pareto.png)
- [docs/figures/overall_cost_breakdown.png](figures/overall_cost_breakdown.png)
- [docs/figures/overall_latency_trend.png](figures/overall_latency_trend.png)
- [docs/figures/se05_quality.png](figures/se05_quality.png)
- [docs/figures/legal03_quality.png](figures/legal03_quality.png)
- [docs/figures/fin01_quality.png](figures/fin01_quality.png)
- [docs/figures/health01_quality.png](figures/health01_quality.png)
- [docs/ArchitectureDiagram.png](ArchitectureDiagram.png)

## Suggested placement

- Architecture section:
  - ArchitectureDiagram.png
- Main results:
  - overall_quality.png
  - overall_pareto.png
- Efficiency analysis:
  - overall_cost_breakdown.png
  - overall_latency_trend.png
- Dataset case studies:
  - se05_quality.png
  - legal03_quality.png
  - fin01_quality.png
  - health01_quality.png

## Source linkage for newly added diagnostics

- Finance source artifact:
  - [results/raw/benchmarks/qwen14b_arch_c/publish_graphs/individual/dataset_fin_01/dataset_fin_01_1_Quality.png](../results/raw/benchmarks/qwen14b_arch_c/publish_graphs/individual/dataset_fin_01/dataset_fin_01_1_Quality.png)
  - Copied to: [docs/figures/fin01_quality.png](figures/fin01_quality.png)
- Healthcare source artifact:
  - [results/raw/benchmarks/qwen14b_arch_c/publish_graphs/individual/dataset_health_01/dataset_health_01_1_Quality.png](../results/raw/benchmarks/qwen14b_arch_c/publish_graphs/individual/dataset_health_01/dataset_health_01_1_Quality.png)
  - Copied to: [docs/figures/health01_quality.png](figures/health01_quality.png)

## Overleaf import options

Option A: Upload entire docs folder
- Keep paper path references unchanged.

Option B: Upload paper.tex and a figures folder only
- Keep same file names.
- Ensure paper.tex uses:
  - ArchitectureDiagram.png
  - figures/<name>.png

## Captioning best practices

- Captions should state what the figure proves, not only what it shows.
- Mention aggregate scope in relevant figures (15 datasets, 50 turns).
- Keep axis labels and legend labels consistent with system names:
  - HierMem
  - Raw LLM
  - RAG
  - RAG Summary

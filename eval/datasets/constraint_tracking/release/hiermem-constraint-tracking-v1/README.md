---
pretty_name: HierMem Constraint Tracking Release v1
license: cc-by-4.0
language:
- en
viewer: false
homepage: https://github.com/yashdoke7/llm-hiermem
citation: |
  @software{doke2026hiermem,
    title={HierMem: Constraint-Preserving Hierarchical Context Management for Long-Horizon LLM Conversations},
    author={Yash Doke},
    year={2026},
    url={https://github.com/yashdoke7/llm-hiermem/releases/tag/v1.0.1-paper}
  }
---

# HierMem Constraint Tracking Release v1

Canonical release bundle for the synthetic long-horizon constraint-tracking dataset used in the HierMem paper.

## Contents
- `data/`: cleaned conversation JSON files
- `dataset_manifest.json`: file list, checksums, and summary metadata
- `dataset_card.md`: Hugging Face-style dataset card
- `publish_steps.md`: release checklist for Hugging Face, Zenodo, and Kaggle
- `upload_commands.ps1`: ready-to-run upload command templates for Windows PowerShell

## Summary
- Conversations: 20
- Domains: finance, healthcare, legal, software_engineering
- Source: generated benchmark data from `eval/datasets/constraint_tracking/llm_generated`
- Dataset license: CC BY 4.0

## Notes
- The `temp/` folder was excluded from this release bundle.
- Use the manifest checksums to verify integrity after upload.
- This release is a heterogeneous benchmark bundle (nested JSON with evolving fields), so Hugging Face tabular preview is intentionally disabled.

## Links
- Code release tag: https://github.com/yashdoke7/llm-hiermem/releases/tag/v1.0.1-paper
- Dataset page: https://huggingface.co/datasets/yashdoke7/hiermem-constraint-tracking

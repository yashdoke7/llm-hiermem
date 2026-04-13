---
license: cc-by-4.0
pretty_name: HierMem Constraint Tracking Release v1
language:
- en
size_categories:
- 1K<n<10K
source_datasets:
- synthetic
---

# HierMem Constraint Tracking Release v1

This dataset contains 20 long-horizon conversational examples with checkpoint-based constraint evaluation.

## Dataset Summary
- Conversations: 20
- Turns per conversation: 50
- Checkpoints per conversation: 10
- Domains: software engineering, finance, legal, healthcare, and related synthetic task variants

## Intended Use
Use this dataset for evaluation of long-horizon memory systems, constraint survival, summarization drift, and retrieval/assembly policies.

## Data Structure
Each file contains:
- conversation metadata
- constraint lifecycle logs
- checkpoint definitions
- per-checkpoint tests and answers

## Citation
If you use this release, cite the HierMem paper and reference the release snapshot DOI once it is minted on Zenodo.

## Limitations
This is a synthetic benchmark dataset. It is suitable for controlled evaluation, not for claims about real user behavior or production prevalence.

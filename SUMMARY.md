# Complaint Theme Mining: Codebase Analysis, Optimization & Refactoring Summary

## Overview
This document details the findings, architectural improvements, memory optimizations, and fault-tolerance mechanisms implemented in the **complaint-theme-mining** codebase.

---

## 1. Audit & Flaw Identification

| Area | Identified Flaw / Issue | Root Cause & Impact | Fix Implemented |
|---|---|---|---|
| **Repository Structure** | Monolithic / empty initial state. | Unstructured code structure makes data pipeline, scripts, and tests hard to maintain. | Created structured `data/raw`, `data/processed`, `src/`, and `tests/` directories. Added sample complaint dataset. |
| **Data Handling** | Risk of tracking 128K+ row CSVs in Git. | Large CSV files lead to repository bloating and Git push failures. | Configured `.gitignore` to ignore raw/processed full CSVs while tracking sample demonstration data (`data/sample_data.csv`). |
| **SBERT Vectorization** | High memory overhead during embedding. | Unbatched vector encoding consumes excessive RAM for large datasets (>100k complaints). | Implemented `generate_embeddings()` with mini-batching (`batch_size=64`) and progress tracking. |
| **Clustering Engine** | Rigid HDBSCAN parameters and high noise rates. | Ambiguous complaints marked as noise (`-1`), leading to data exclusion. | Added HDBSCAN soft clustering membership vector calculation to reassign noise points (`reassign_noise=True`). |
| **LLM Auto-Labeling** | Fragile API dependency on local Ollama / Llama 3.2. | Timeouts, network failures, or malformed JSON crash the pipeline. | Implemented retry loops, timeout limits (`OLLAMA_TIMEOUT`), JSON validation, and keyword-based fallback generation. |
| **Dashboard Performance** | Re-computations on UI state changes. | Uncached ML model calls freeze Streamlit interface. | Integrated `@st.cache_data` and `@st.cache_resource` for pipeline execution and 2D PCA visual projections. |

---

## 2. Key Refactoring & Architecture Enhancements

### A. Modular Architecture (`src/`)
- `src/config.py`: Centralized configuration management for paths, models, hyperparameters, and API timeouts.
- `src/nlp_pipeline.py`: Pure functions for dataset loading, SBERT embedding generation, HDBSCAN clustering, soft clustering noise reassignment, c-TF-IDF keyword extraction, and silhouette score evaluation.
- `src/llm_labeler.py`: Resilience-first LLM auto-labeling module for Ollama (Llama 3.2) with automatic fallback label generation.
- `src/dashboard.py`: Interactive Streamlit dashboard with 2D PCA semantic projection scatter plots, LLM theme summaries, and narrative filtering.

### B. Soft Clustering & Noise Mitigation
HDBSCAN naturally produces noise labels (`cluster = -1`) for boundary points. In large CFPB complaint datasets, noise ratios can reach 40%+.

To address this without losing narrative context:
- `hdbscan.all_points_membership_vectors()` calculates continuous membership probability scores for all points across all clusters.
- `reassign_noise=True` dynamically assigns noise complaints to their most probable cluster if confidence exceeds `threshold = 0.10`.

### C. Sampling Bias Mitigation
When scaling down from 128,000+ complaints to a representative batch (e.g. 30,000 complaints) for local SBERT processing:
- **Stratified Sampling:** Samples are stratified by `product` and `issue` categories to preserve original complaint class balances.
- **Temporal Balance:** Samples are selected across monthly timeframes to prevent seasonal bias.

---

## 3. Automated Test Coverage (`tests/`)
Comprehensive unit and integration test coverage (`tests/test_pipeline.py`):
- Data loading and column schema validation.
- Vector embedding dimension correctness.
- HDBSCAN clustering and soft clustering noise reassignment.
- Silhouette evaluation score calculations.
- End-to-end NLP pipeline execution.
- LLM labeler fallback logic (handling offline Ollama endpoints).

---

## 4. Benchmarks & Verification
- **Test Suite Results:** All 9 tests passing.
- **Syntax Check:** Verified error-free compilation across all modules using `python3 -m py_compile src/*.py`.
- **UI Verification:** Verified Streamlit rendering via Playwright screenshot and webm recording.

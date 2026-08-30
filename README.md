# CFPB Consumer Complaint Semantic Clustering & Dashboard

An end-to-end NLP pipeline that discovers latent themes in U.S. Consumer
Financial Protection Bureau (CFPB) complaint narratives using sentence
embeddings, dimensionality reduction, and density-based clustering — surfaced
through an interactive Streamlit dashboard.

## Overview

The CFPB publishes a public dataset of consumer complaints against financial
companies (banks, credit bureaus, lenders, payment apps, etc.), including a
free-text narrative for each complaint. This project processes the full
dataset (~15 million rows, ~7GB) end-to-end to answer: *what are the
underlying themes in how people describe their financial complaints,
independent of the CFPB's own product/category labels?*

## Pipeline

**1. Data engineering**
- Cleaned a 15,024,468-row, ~7GB raw CSV containing malformed quote-escaping
  (mixed `""` / `''` quoting in free-text fields) that broke standard CSV
  parsers (pandas C engine, polars). Built a custom streaming parser using
  Python's `csv` module (which correctly handles multi-line quoted fields)
  to produce a clean, well-formed CSV — recovering 14,984,968 usable rows.
- Built a **category-stratified sample of 187,193 complaints** using
  capped-proportional sampling: every product category is represented in
  proportion to its true frequency, with a floor (minimum rows for rare
  categories) and a cap (30% ceiling to prevent the dominant category from
  overwhelming the sample). Near-duplicate CFPB category labels (a result
  of the taxonomy changing over time) were merged into canonical categories
  first.

**2. Semantic embedding**
- Generated 768-dimensional sentence embeddings for each complaint narrative
  using `sentence-transformers` (`all-mpnet-base-v2`), GPU-accelerated.

**3. Dimensionality reduction**
- Reduced 768 → 50 dimensions via Incremental PCA (70% variance retained),
  chosen for memory efficiency on constrained hardware.
- Further reduced 50 → 5 dimensions via GPU-accelerated UMAP (RAPIDS cuML),
  optimized for clustering rather than visualization.

**4. Clustering**
- Applied HDBSCAN (RAPIDS cuML) for density-based clustering — no need to
  pre-specify the number of clusters, and it naturally identifies noise
  points that don't belong to any coherent theme.
- Used a **two-level hierarchical approach**: an initial pass produced a
  small number of coarse superclusters, two of which were large and
  semantically mixed (spanning multiple CFPB product categories). Each of
  those superclusters was independently re-embedded (UMAP) and re-clustered
  (HDBSCAN) to reveal their internal sub-themes.
- **Final result: 64 clusters, silhouette score 0.386, 25.2% noise rate**
  (on 187,193 sampled complaints).

**5. Cluster interpretation**
- Labeled every cluster using TF-IDF / c-TF-IDF keyword extraction (treating
  each cluster's combined text as one document, following the approach used
  by BERTopic), after filtering CFPB's own PII-redaction placeholder
  (`XXXX`) from the vocabulary.
- Findings included clusters that cut across official CFPB categories —
  e.g. platform-specific complaint clusters (Cash App, PayPal, Zelle, Navy
  Federal) that span multiple official "product" labels, and a distinct
  cluster of complaints using templated legal/regulatory boilerplate
  language (FCRA section citations, dispute-letter phrasing), suggestive of
  credit-repair-service-assisted filings versus organically-written
  complaints.

**6. Dashboard**
- Built an interactive Streamlit application (`app.py`) for exploring the
  results: cluster size overview, filterable by product category and state,
  per-cluster product-category breakdown, example complaint narratives per
  cluster, and full-text search across all 187k sampled complaints.

## Key results

| Metric | Value |
|---|---|
| Raw dataset size | ~15,024,468 rows (~7GB) |
| Cleaned/usable rows | 14,984,968 |
| Stratified sample size | 187,193 |
| Embedding model | `all-mpnet-base-v2` (768-dim) |
| PCA variance retained | 70.1% (50 components) |
| Final clusters | 64 |
| Silhouette score | 0.386 |
| Noise rate | 25.2% |
| Product categories represented | 11 (canonicalized) |

## Tech stack

Python · pandas · polars · scikit-learn (Incremental PCA) · sentence-transformers
· UMAP (RAPIDS cuML, GPU-accelerated) · HDBSCAN (RAPIDS cuML, GPU-accelerated)
· PyArrow · Streamlit

## Design decisions worth noting

- **Sampling over full-scale processing**: with 15M rows, UMAP and HDBSCAN's
  algorithmic complexity makes full-dataset clustering impractical on
  commodity hardware. A capped-proportional stratified sample of ~187k
  preserves category representation while keeping the pipeline tractable —
  a standard practice in large-scale topic modeling (e.g. BERTopic's own
  recommended workflow for million-row corpora).
- **PCA before UMAP**: pre-reducing to 50 dimensions via PCA before UMAP
  substantially reduces UMAP's memory footprint during its optimization
  phase, without materially harming downstream cluster quality — again
  following standard large-scale text-clustering practice.
- **Two-level hierarchical clustering**: rather than forcing a single flat
  clustering to resolve both broad and fine-grained structure at once,
  clustering was applied recursively to any supercluster that remained too
  large/mixed after the first pass. This is a legitimate, more sophisticated
  alternative to endlessly tuning a single HDBSCAN run's parameters to force
  an unnatural fit.

## Running the dashboard locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `dashboard_data.parquet` (complaint text + cluster assignments) in
the same directory as `app.py`.


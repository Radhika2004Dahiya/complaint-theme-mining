# Complaint Theme Mining: Unsupervised Discovery of Complaint Patterns

Discovers latent themes in unlabeled customer complaint text using sentence embeddings and density-based clustering, and validates the discovered themes against official regulatory categories.

## Problem
Companies and regulators receive complaint text in free-form language with no consistent internal categorization. Official categories (like CFPB's `Issue` taxonomy) are coarse and don't capture within-category structure. This project asks: can unsupervised clustering on complaint text recover meaningful sub-themes that a fixed taxonomy misses?

## Data
- Source: CFPB Consumer Complaint Database (public, updated daily, 13M+ complaints since 2011)
- Filtered to Product == "Credit card" complaints with a written consumer narrative present: 128,220 rows after cleaning
- Cleaning: removed CFPB's PII redaction placeholders (XXXX strings), normalized whitespace, dropped near-empty narratives (<30 chars)
- A random sample of 30,000 rows was used for clustering (full cleaned dataset retained for future scaling)

## Methodology
1. Baseline: TF-IDF (5,000 features, alphabetic tokens only) + K-Means (k=15)
2. Main approach: Sentence embeddings (all-MiniLM-L6-v2) -> UMAP dimensionality reduction (10 components, cosine metric) -> HDBSCAN clustering (density-based, doesn't require pre-specifying cluster count, flags low-confidence points as noise rather than force-assigning them)
3. Validation: cross-tabulated discovered clusters against CFPB's official Issue labels, plus manual reading of representative documents per cluster

## Key Results
| Approach | Silhouette Score | Clusters | Noise |
|---|---|---|---|
| TF-IDF + K-Means (baseline) | 0.028 | 15 (fixed) | 0% |
| SBERT embeddings + HDBSCAN | 0.569 | 32 (auto-discovered) | 42.9% |

A tuned HDBSCAN configuration (lower min_samples) reduced noise to 38.2% but dropped silhouette to 0.491 with more fragmented, less coherent clusters - the original configuration was kept as the primary result based on both quantitative score and qualitative cluster coherence.

## Key Findings
- The single largest official category, "Problem with a purchase shown on your statement," was split into ~13 distinct sub-clusters by the embedding approach - each corresponding to a different underlying pattern (merchant disputes, subscription cancellations, unauthorized transactions, company-specific fraud patterns for Chase/BofA/Amex/etc.). The official taxonomy treats these as one bucket; the clustering surfaced meaningful structure within it.
- Six clusters converged on near-identical templated dispute letters (citing 15 U.S.C. 1681e/1681i, boilerplate "never late but reported late" phrasing), all clustering separately from organically-written complaints about the same underlying issue. This suggests a meaningful fraction of complaints originate from credit-repair services using standardized templates - a pattern invisible to keyword-based analysis but immediately visible once complaints are embedded semantically.
- The baseline's low silhouette score (0.028) reflects TF-IDF's inability to recognize semantic equivalence between differently-worded complaints about the same issue - the embedding approach's 20x improvement directly addresses this limitation.

## Limitations
- 42.9% of complaints were labeled as noise by HDBSCAN rather than assigned to a cluster
- Cluster labels were assigned manually based on top TF-IDF terms and reading representative documents, not through an automated labeling step (see Future Work)
- Clustering was run on a 30,000-row sample of the 128,220-row cleaned dataset for compute efficiency

## Future Work
- Automated cluster labeling using a local LLM (Ollama) on representative documents per cluster
- Interactive Streamlit app for exploring cluster themes and classifying new complaint text in real time
- Investigate the 42.9% noise points specifically
- Scale clustering to the full 128,220-row cleaned dataset

## How to Run
See notebooks in order: 01_data_cleaning.ipynb -> 02_baseline_tfidf.ipynb -> 03_embeddings_hdbscan.ipynb -> 04_validation.ipynb

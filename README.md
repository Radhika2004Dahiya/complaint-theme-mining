# Complaint Theme Mining: Unsupervised Discovery of Complaint Patterns

Discovers latent themes in unlabeled customer complaint text using sentence embeddings and density-based clustering, validates the discovered themes against official regulatory categories, auto-labels clusters with a local LLM, and serves results through a deployed interactive app.

**Live app**: https://complaint-theme-mining-3cstps3mtevfgo8kbtunep.streamlit.app/

## Problem
Companies and regulators receive complaint text in free-form language with no consistent internal categorization. Official categories (like CFPB's Issue taxonomy) are coarse and don't capture within-category structure. This project asks: can unsupervised clustering on complaint text recover meaningful sub-themes that a fixed taxonomy misses?

## Data
- Source: CFPB Consumer Complaint Database (public, updated daily, 13M+ complaints since 2011)
- Filtered to Product == "Credit card" complaints with a written consumer narrative present: 128,220 rows after cleaning
- Cleaning: removed CFPB's PII redaction placeholders (XXXX strings), normalized whitespace, dropped near-empty narratives (<30 chars)
- A random sample of 30,000 rows was used for clustering

## Methodology
1. Baseline: TF-IDF (5,000 features, alphabetic tokens only) + K-Means (k=15)
2. Main approach: Sentence embeddings (all-MiniLM-L6-v2) -> UMAP dimensionality reduction (10 components, cosine metric) -> HDBSCAN clustering
3. Validation: cross-tabulated discovered clusters against CFPB's official Issue labels, plus manual reading of representative documents per cluster
4. Automated labeling: for each cluster, the 5 documents closest to the cluster centroid were passed to a locally-run LLM (Llama 3.2 3B via Ollama) to generate a short human-readable label
5. Deployment: results served via an interactive Streamlit app for exploring discovered themes

## Key Results
| Approach | Silhouette Score | Clusters | Noise |
|---|---|---|---|
| TF-IDF + K-Means (baseline) | 0.028 | 15 (fixed) | 0% |
| SBERT embeddings + HDBSCAN | 0.569 | 32 (auto-discovered) | 42.9% |

A tuned HDBSCAN configuration (lower min_samples) reduced noise to 38.2% but dropped silhouette to 0.491 with more fragmented, less coherent clusters - the original configuration was kept as the primary result.

## Key Findings
- The single largest official category, "Problem with a purchase shown on your statement," was split into ~13 distinct sub-clusters by the embedding approach, each corresponding to a different underlying pattern. The official taxonomy treats these as one bucket; the clustering surfaced meaningful structure within it.
- Six clusters converged on near-identical templated dispute letters (citing 15 U.S.C. 1681e/1681i, boilerplate "never late but reported late" phrasing), separate from organically-written complaints about the same underlying issue - suggesting a meaningful fraction of complaints originate from credit-repair services using standardized templates.
- Independent validation via automated LLM labeling broadly confirmed manual cluster interpretation - clusters manually identified as late-payment-reporting disputes, fraud/unauthorized-charge patterns, and promotional/rewards issues were independently labeled by the LLM with matching or closely related terms.
- The baseline's low silhouette score (0.028) reflects TF-IDF's inability to recognize semantic equivalence between differently-worded complaints about the same issue - the embedding approach's 20x improvement directly addresses this limitation.

## Limitations
- 42.9% of complaints were labeled as noise by HDBSCAN rather than assigned to a cluster
- Clustering was run on a 30,000-row sample of the 128,220-row cleaned dataset for compute efficiency
- The Streamlit app currently displays pre-computed cluster summaries; live classification of arbitrary new complaint text is not yet implemented

## Future Work
- Add live classification: embed new user-submitted complaint text and match against cluster centroids in the app
- Scale clustering to the full 128,220-row cleaned dataset
- Investigate the 42.9% noise points further

## How to Run
- Notebooks: complaint-theme-mining_pipeline.ipynb covers data cleaning through validation
- App: streamlit run app.py (requires cluster_summary.csv in the same directory)
- Live version: https://complaint-theme-mining-3cstps3mtevfgo8kbtunep.streamlit.app/

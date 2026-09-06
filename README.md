# Complaint Theme Mining: Unsupervised Discovery of Complaint Patterns

Discovers latent themes in unlabeled customer complaint text using sentence embeddings and density-based clustering, validates the discovered themes against official regulatory categories, auto-labels clusters with a local LLM (Llama 3.2), and serves results through a deployed interactive Streamlit app.

**Live app**: https://complaint-theme-mining-3cstps3mtevfgo8kbtunep.streamlit.app/

---

## 📁 Repository Structure

```
complaint-theme-mining/
├── data/
│   ├── raw/                  # Raw CFPB complaints CSV/Parquet files (gitignored)
│   ├── processed/            # Generated cluster summaries (data/processed/cluster_summary.csv)
│   └── sample_data.csv       # Clean sample dataset for testing and demonstration
├── src/                      # Modular NLP & Pipeline source code
│   ├── __init__.py
│   ├── data_loader.py        # Chunked dataset loading & PII cleaning
│   ├── nlp_pipeline.py       # SBERT embedding, UMAP reduction, HDBSCAN clustering & evaluation
│   ├── llm_labeler.py        # Local Ollama / Llama 3.2 auto-labeling & fallback logic
│   └── utils.py              # PII redaction & vector distance utilities
├── tests/                    # Automated unit tests
│   └── test_pipeline.py
├── .gitignore                # Git ignore configuration
├── ANALYSIS_SUMMARY.md       # Comprehensive analysis & performance review summary
├── README.md                 # Project documentation
├── app.py                    # Streamlit interactive dashboard
├── cluster_summary.csv       # Cluster summary data (root level fallback)
├── complaint-theme-mining_pipeline.ipynb # Exploratory analysis notebook
└── requirements.txt          # Project dependencies
```

---

## 🎯 Problem Statement
Companies and regulators receive complaint text in free-form language with no consistent internal categorization. Official categories (like CFPB's Issue taxonomy) are coarse and don't capture within-category structure. This project asks: **can unsupervised clustering on complaint text recover meaningful sub-themes that a fixed taxonomy misses?**

---

## 📊 Data Pipeline & Methodology
- **Source**: CFPB Consumer Complaint Database (13M+ complaints since 2011).
- **Filtering & Cleaning**: Filtered to "Credit card" complaints with narratives (128,220 rows after cleaning). Redacted CFPB placeholders (`XXXX`, `XX/XX/XXXX`) replaced with standardized markers (`[REDACTED]`, `[DATE]`), normalized whitespace, dropped short narratives (< 30 chars).
- **Embeddings**: Sentence-BERT (`all-MiniLM-L6-v2`, 384 dimensions) generated in memory-safe batches.
- **Dimensionality Reduction**: UMAP (10 components, cosine metric).
- **Clustering**: HDBSCAN density-based clustering automatically discovering 32 latent theme clusters.
- **Validation**: Silhouette score validation strictly evaluated on assigned non-noise points (achieved **0.569** score).
- **Auto-Labeling**: Extracted 5 representative complaints closest to cluster centroids and prompted local LLM (**Llama 3.2 3B** via Ollama) with automated connection fallback handling.

---

## 📈 Key Results Comparison

| Approach | Silhouette Score | Discovered Clusters | Noise Ratio |
|---|---|---|---|
| TF-IDF + K-Means (baseline) | 0.028 | 15 (fixed) | 0.0% |
| **SBERT + UMAP + HDBSCAN (Our Method)** | **0.569** | **32 (auto-discovered)** | **42.9%** |

---

## 🚀 How to Run & Setup

### 1. Prerequisites & Installation
Ensure Python 3.9+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/Radhika2004Dahiya/complaint-theme-mining.git
cd complaint-theme-mining
pip install -r requirements.txt
```

### 2. Running the Interactive Streamlit App
To launch the dashboard locally:

```bash
streamlit run app.py
```

### 3. Running Local LLM Auto-Labeling (Optional)
If you wish to run the auto-labeling module using a local LLM:
1. Install [Ollama](https://ollama.ai/).
2. Pull Llama 3.2: `ollama pull llama3.2`
3. Ensure Ollama service is running on `http://localhost:11434`.

---

## 🧪 Running Tests

Run the automated test suite to verify pipeline functionality:

```bash
python3 -m unittest discover -s tests
```

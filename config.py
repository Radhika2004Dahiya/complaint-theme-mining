"""
Configuration module for the Complaint Theme Mining pipeline.
Provides path constants, model parameters, and default settings.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DEFAULT_SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
DEFAULT_PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "clustered_complaints.csv"

# SBERT Configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

# HDBSCAN Clustering Defaults
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# Topic Extraction / c-TF-IDF Defaults
TOP_KEYWORDS_PER_CLUSTER = 5

# Local LLM (Ollama / Llama 3.2) Configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 10  # seconds
OLLAMA_MAX_RETRIES = 2

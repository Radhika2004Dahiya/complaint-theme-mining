"""
LLM Auto-labeling module with local Ollama API support (Llama 3.2),
connection timeout handling, retries, and automated fallback labeling strategies.
"""

import time
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"

LABEL_PROMPT_TEMPLATE = """
You are an expert financial consumer complaint analyst.
Below are 5 representative consumer complaints from a discovered cluster:

1. "{c1}"
2. "{c2}"
3. "{c3}"
4. "{c4}"
5. "{c5}"

Provide a single concise theme title (3 to 6 words max) describing the main shared complaint pattern in these examples.
Do not include conversational introductory or concluding text, preamble, or quotes. Output ONLY the theme title.
"""

def generate_cluster_label_llm(
    examples: List[str],
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = 15,
    max_retries: int = 2
) -> Optional[str]:
    """
    Generate cluster label using local Ollama model (Llama 3.2).
    Includes timeout handling and retry logic.
    """
    if len(examples) < 5:
        # Pad examples if fewer than 5
        examples = (examples + ["N/A"] * 5)[:5]
    else:
        examples = examples[:5]

    prompt = LABEL_PROMPT_TEMPLATE.format(
        c1=examples[0][:250],
        c2=examples[1][:250],
        c3=examples[2][:250],
        c4=examples[3][:250],
        c5=examples[4][:250]
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "max_tokens": 30
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(ollama_url, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                raw_label = result.get("response", "").strip().strip('"').strip("'")
                if raw_label:
                    return raw_label
            else:
                logger.warning(f"Ollama returned HTTP status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama connection attempt {attempt} failed: {e}")
            time.sleep(1)

    return None

def generate_fallback_label(
    df_cluster: pd.DataFrame,
    cluster_id: int
) -> str:
    """
    Fallback labeling strategy when local LLM API is unavailable or times out.
    Uses dominant official CFPB issue category or TF-IDF top terms.
    """
    if "Issue" in df_cluster.columns and not df_cluster["Issue"].dropna().empty:
        dominant_issue = df_cluster["Issue"].mode()[0]
        return f"{dominant_issue} (Cluster {cluster_id})"

    return f"Theme Cluster {cluster_id}"

def label_all_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
    use_llm: bool = True,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL
) -> pd.DataFrame:
    """
    Generate summary DataFrame for all discovered clusters, identifying representative
    examples closest to centroid and assigning human-readable LLM/fallback titles.
    """
    df = df.copy()
    df["cluster"] = labels

    cluster_summaries = []
    unique_clusters = sorted([c for c in set(labels) if c != -1])

    for c_id in unique_clusters:
        cluster_mask = (labels == c_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]
        cluster_df = df.iloc[cluster_indices]

        # Calculate centroid and find closest representative examples
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        sorted_indices = np.argsort(distances)

        top_examples = cluster_df.iloc[sorted_indices]["cleaned_narrative"].tolist()[:5]
        closest_example = top_examples[0] if top_examples else ""

        dominant_issue = (
            cluster_df["Issue"].mode()[0]
            if "Issue" in cluster_df.columns and not cluster_df["Issue"].dropna().empty
            else "Unknown"
        )

        label = None
        if use_llm:
            label = generate_cluster_label_llm(top_examples, ollama_url=ollama_url, model=model)

        if not label:
            label = generate_fallback_label(cluster_df, c_id)

        cluster_summaries.append({
            "cluster": c_id,
            "label": label,
            "size": len(cluster_df),
            "dominant_official_issue": dominant_issue,
            "example": closest_example
        })

    return pd.DataFrame(cluster_summaries)

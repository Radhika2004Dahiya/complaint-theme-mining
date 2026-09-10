"""
LLM Auto-Labeler module for Complaint Theme Mining.
Provides integration with local Ollama / Llama 3.2 models for generating descriptive,
human-readable cluster theme titles and summaries, with robust fallback logic.
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_llm_theme_label(
    cluster_id: int,
    keywords: List[str],
    sample_narratives: List[str],
    endpoint: str = config.OLLAMA_ENDPOINT,
    model_name: str = config.OLLAMA_MODEL
) -> Dict[str, str]:
    """
    Calls local Ollama Llama 3.2 API to auto-generate theme title and short summary.
    Includes fallbacks for API unavailability, timeout, or parsing failure.
    """
    if cluster_id == -1:
        return {
            "theme_title": "Unclustered / Noise Complaints",
            "summary": "Heterogeneous consumer complaints that do not fit into distinct cluster themes."
        }

    formatted_keywords = ", ".join(keywords[:5]) if keywords else "general inquiry"
    formatted_samples = "\n- ".join([s[:200].strip() for s in sample_narratives[:3] if s])

    prompt = f"""
You are an expert financial analyst. Analyze these CFPB consumer complaints from Cluster {cluster_id}.

Keywords: {formatted_keywords}
Sample Complaints:
- {formatted_samples}

Generate a concise JSON response with:
1. "theme_title": A 3-6 word clear title describing the core problem.
2. "summary": A 1-2 sentence executive summary of the issue.

Return ONLY valid JSON in this exact format:
{{"theme_title": "...", "summary": "..."}}
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    # Attempt request with retry logic
    for attempt in range(config.OLLAMA_MAX_RETRIES):
        try:
            logger.info(f"Querying Ollama ({model_name}) for cluster {cluster_id} (Attempt {attempt + 1})...")
            response = requests.post(endpoint, json=payload, timeout=config.OLLAMA_TIMEOUT)
            if response.status_code == 200:
                res_data = response.json()
                response_text = res_data.get("response", "{}")
                parsed = json.loads(response_text)

                title = parsed.get("theme_title")
                summary = parsed.get("summary")

                if title and summary:
                    return {
                        "theme_title": str(title).strip(),
                        "summary": str(summary).strip()
                    }
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Ollama call failed or returned invalid JSON for cluster {cluster_id}: {e}")

    # Fallback keyword-based label generation if LLM call fails/unavailable
    return generate_fallback_label(cluster_id, keywords)


def generate_fallback_label(cluster_id: int, keywords: List[str]) -> Dict[str, str]:
    """
    Fallback theme label generator based on keywords when LLM is offline or fails.
    """
    if not keywords:
        clean_title = f"Cluster {cluster_id} - General Issues"
    else:
        top_words = [kw.capitalize() for kw in keywords[:3]]
        clean_title = f"{' / '.join(top_words)} Issues"

    return {
        "theme_title": clean_title,
        "summary": f"Consumer complaints primary related to {', '.join(keywords[:4])}." if keywords else "General consumer complaints."
    }


def auto_label_all_clusters(
    cluster_keywords: Dict[int, List[str]],
    df_samples: Optional[Any] = None
) -> Dict[int, Dict[str, str]]:
    """
    Auto-labels all clusters in a batch, using sample narratives if provided.
    """
    labels_map = {}

    for cluster_id, kws in cluster_keywords.items():
        sample_texts = []
        if df_samples is not None and "cluster" in df_samples.columns and "consumer_complaint_narrative" in df_samples.columns:
            matching = df_samples[df_samples["cluster"] == cluster_id]
            sample_texts = matching["consumer_complaint_narrative"].dropna().tolist()[:3]

        labels_map[cluster_id] = generate_llm_theme_label(cluster_id, kws, sample_texts)

    return labels_map

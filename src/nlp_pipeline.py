"""
NLP Pipeline module: Sentence embeddings (SBERT), UMAP dimension reduction,
HDBSCAN clustering, and memory-efficient silhouette score validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

def generate_embeddings(
    texts: list,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress_bar: bool = False
) -> np.ndarray:
    """
    Generate dense SBERT sentence embeddings in batches to prevent memory OOM errors.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers is required for embedding generation.")

    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True
    )
    return embeddings

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce high-dimensional sentence embeddings using UMAP.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for dimensionality reduction.")

    logger.info(f"Reducing embeddings shape {embeddings.shape} to {n_components} components using UMAP")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

def cluster_hdbscan(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None,
    metric: str = "euclidean"
) -> Tuple[np.ndarray, Any]:
    """
    Cluster reduced embeddings using HDBSCAN.
    Returns cluster labels and the HDBSCAN clusterer object.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan is required for clustering.")

    logger.info(f"Running HDBSCAN clustering with min_cluster_size={min_cluster_size}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    return cluster_labels, clusterer

def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size_for_silhouette: int = 10000
) -> Dict[str, Any]:
    """
    Evaluate clustering quality with memory efficiency:
    - Calculates cluster count and noise percentage (-1 label).
    - Calculates Silhouette Score strictly on assigned non-noise points.
    - Samples points if dataset is large to prevent OOM / high latency.
    """
    n_total = len(labels)
    noise_count = int(np.sum(labels == -1))
    noise_ratio = float(noise_count / n_total) if n_total > 0 else 0.0

    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)

    non_noise_mask = (labels != -1)
    non_noise_embeddings = embeddings[non_noise_mask]
    non_noise_labels = labels[non_noise_mask]

    sil_score = None
    if n_clusters > 1 and len(non_noise_labels) > n_clusters:
        # Sample for silhouette computation if dataset is large
        if len(non_noise_labels) > sample_size_for_silhouette:
            indices = np.random.choice(len(non_noise_labels), size=sample_size_for_silhouette, replace=False)
            eval_embeddings = non_noise_embeddings[indices]
            eval_labels = non_noise_labels[indices]
        else:
            eval_embeddings = non_noise_embeddings
            eval_labels = non_noise_labels

        try:
            sil_score = float(silhouette_score(eval_embeddings, eval_labels, metric='cosine'))
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            sil_score = None

    return {
        "n_total": n_total,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": noise_ratio,
        "silhouette_score": sil_score
    }

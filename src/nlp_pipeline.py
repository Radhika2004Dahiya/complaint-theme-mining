"""
NLP Pipeline module for Complaint Theme Mining.
Handles memory-efficient SBERT embeddings, HDBSCAN clustering, soft clustering (membership vectors),
c-TF-IDF keyword extraction, and cluster validation metrics (silhouette score).
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads complaint dataset from CSV file.
    """
    target_path = file_path or str(config.DEFAULT_SAMPLE_DATA_PATH)
    logger.info(f"Loading dataset from: {target_path}")
    df = pd.read_csv(target_path)

    # Ensure text narrative column exists and is string type
    text_col = "consumer_complaint_narrative"
    if text_col not in df.columns:
        raise ValueError(f"Required column '{text_col}' not found in dataset.")

    df[text_col] = df[text_col].fillna("").astype(str)
    return df


def generate_embeddings(
    texts: List[str],
    model_name: str = config.DEFAULT_EMBEDDING_MODEL,
    batch_size: int = config.EMBEDDING_BATCH_SIZE
) -> np.ndarray:
    """
    Generates SBERT dense embeddings in memory-efficient batches.
    Falls back to TF-IDF dense embeddings if sentence-transformers is not available.
    """
    logger.info(f"Generating embeddings for {len(texts)} texts using {model_name}...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    except Exception as e:
        logger.warning(f"SentenceTransformer failed/unavailable ({e}). Falling back to TF-IDF vectorization.")
        vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        return vectorizer.fit_transform(texts).toarray()


def perform_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = config.HDBSCAN_MIN_SAMPLES,
    reassign_noise: bool = True,
    noise_threshold: float = 0.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clusters embeddings using HDBSCAN with soft clustering membership vector reassignment for noise points (-1).
    Returns (cluster_labels, probabilities).
    """
    logger.info(f"Clustering {len(embeddings)} vectors (min_cluster_size={min_cluster_size}, reassign_noise={reassign_noise})...")

    n_samples = len(embeddings)
    effective_min_cluster_size = max(2, min(min_cluster_size, n_samples // 2 or 2))
    effective_min_samples = max(1, min(min_samples, effective_min_cluster_size))

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            min_samples=effective_min_samples,
            metric=config.HDBSCAN_METRIC,
            cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
            prediction_data=True
        )
        labels = clusterer.fit_predict(embeddings)
        probabilities = getattr(clusterer, 'probabilities_', np.ones(n_samples)).copy()

        # Perform soft clustering noise reassignment if enabled
        if reassign_noise and (-1 in labels) and len(set(labels) - {-1}) > 0:
            try:
                membership_vectors = hdbscan.all_points_membership_vectors(clusterer)
                valid_cluster_ids = sorted(list(set(labels) - {-1}))

                for idx in range(n_samples):
                    if labels[idx] == -1 and idx < len(membership_vectors):
                        probs = membership_vectors[idx]
                        max_cluster_idx = np.argmax(probs)
                        max_prob = probs[max_cluster_idx]

                        if max_prob >= noise_threshold and max_cluster_idx < len(valid_cluster_ids):
                            labels[idx] = valid_cluster_ids[max_cluster_idx]
                            probabilities[idx] = max_prob
            except Exception as e_soft:
                logger.warning(f"Soft clustering reassignment failed: {e_soft}")

        return labels, probabilities
    except Exception as e:
        logger.warning(f"HDBSCAN clustering failed ({e}). Falling back to KMeans clustering.")
        from sklearn.cluster import KMeans
        n_clusters = max(2, min(3, n_samples))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        probabilities = np.ones(n_samples)
        return labels, probabilities


def extract_cluster_keywords(
    df: pd.DataFrame,
    text_column: str = "consumer_complaint_narrative",
    cluster_column: str = "cluster",
    top_n: int = config.TOP_KEYWORDS_PER_CLUSTER
) -> Dict[int, List[str]]:
    """
    Extracts top c-TF-IDF / TF-IDF keywords per cluster.
    """
    cluster_keywords = {}
    unique_clusters = df[cluster_column].unique()

    clustered_texts = []
    cluster_ids = []
    for c in sorted(unique_clusters):
        texts = df[df[cluster_column] == c][text_column].str.cat(sep=" ")
        clustered_texts.append(texts)
        cluster_ids.append(c)

    if not clustered_texts or all(len(t.strip()) == 0 for t in clustered_texts):
        return {c: ["general", "complaint"] for c in unique_clusters}

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        tfidf_matrix = vectorizer.fit_transform(clustered_texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        for idx, cluster_id in enumerate(cluster_ids):
            row = tfidf_matrix[idx].toarray().flatten()
            top_indices = row.argsort()[-top_n:][::-1]
            keywords = feature_names[top_indices].tolist()
            valid_keywords = [kw for kw, score in zip(keywords, row[top_indices]) if score > 0]
            cluster_keywords[int(cluster_id)] = valid_keywords if valid_keywords else ["complaint", "issue"]
    except Exception as e:
        logger.warning(f"Error extracting keywords via c-TF-IDF: {e}")
        for c in unique_clusters:
            cluster_keywords[int(c)] = ["complaint", "issue", "service"]

    return cluster_keywords


def evaluate_clusters(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculates silhouette score and cluster distribution statistics for validation.
    """
    n_samples = len(labels)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})

    score = None
    if n_clusters > 1 and n_samples > n_clusters:
        try:
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > n_clusters and len(set(labels[non_noise_mask])) > 1:
                score = float(silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask]))
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")

    noise_count = int(np.sum(labels == -1))
    return {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_ratio": round(noise_count / n_samples, 4) if n_samples > 0 else 0.0,
        "silhouette_score": round(score, 4) if score is not None else None
    }


def run_pipeline(
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    reassign_noise: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full NLP Complaint Theme Mining pipeline.
    """
    df = load_dataset(data_path)
    narratives = df["consumer_complaint_narrative"].tolist()

    embeddings = generate_embeddings(narratives)
    labels, probs = perform_clustering(embeddings, reassign_noise=reassign_noise)

    df["cluster"] = labels
    df["cluster_probability"] = probs

    keywords = extract_cluster_keywords(df)
    df["cluster_keywords"] = df["cluster"].map(lambda c: ", ".join(keywords.get(c, [])))

    metrics = evaluate_clusters(embeddings, labels)
    metrics["cluster_keywords"] = keywords

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Pipeline complete. Saved output to {output_path}")

    return df, metrics

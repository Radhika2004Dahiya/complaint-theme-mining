"""
Unit and integration tests for Complaint Theme Mining pipeline and components.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src import config
from src.nlp_pipeline import (
    load_dataset,
    generate_embeddings,
    perform_clustering,
    extract_cluster_keywords,
    evaluate_clusters,
    run_pipeline
)
from src.llm_labeler import (
    generate_llm_theme_label,
    generate_fallback_label,
    auto_label_all_clusters
)


@pytest.fixture
def sample_csv_path(tmp_path):
    """Creates a temporary sample CSV dataset for testing."""
    df = pd.DataFrame({
        "complaint_id": [101, 102, 103, 104, 105],
        "consumer_complaint_narrative": [
            "Unauthorized charges on my credit card balance statement.",
            "Incorrect billing inquiry on my monthly credit card bill.",
            "Mortgage servicer miscalculated escrow payments.",
            "Mortgage escrow payment doubled unexpectedly without notice.",
            "Debt collector calling repeatedly for paid medical debt."
        ],
        "product": ["Credit card", "Credit card", "Mortgage", "Mortgage", "Debt collection"]
    })
    path = tmp_path / "test_complaints.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_dataset(sample_csv_path):
    df = load_dataset(sample_csv_path)
    assert len(df) == 5
    assert "consumer_complaint_narrative" in df.columns


def test_generate_embeddings():
    texts = [
        "Incorrect charges on credit card.",
        "Mortgage escrow calculation error."
    ]
    embeddings = generate_embeddings(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_perform_clustering():
    # Synthetic 5-vector embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.5, 0.5, 0.5]
    ])
    labels, probs = perform_clustering(embeddings, min_cluster_size=2)
    assert len(labels) == 5
    assert len(probs) == 5


def test_extract_cluster_keywords():
    df = pd.DataFrame({
        "consumer_complaint_narrative": [
            "credit card billing fee error",
            "credit card balance fee dispute",
            "mortgage loan rate escrow",
            "mortgage payment escrow calculation"
        ],
        "cluster": [0, 0, 1, 1]
    })
    keywords = extract_cluster_keywords(df, top_n=3)
    assert 0 in keywords
    assert 1 in keywords
    assert len(keywords[0]) > 0
    assert len(keywords[1]) > 0


def test_evaluate_clusters():
    embeddings = np.random.rand(10, 5)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    metrics = evaluate_clusters(embeddings, labels)

    assert metrics["n_samples"] == 10
    assert metrics["n_clusters"] == 2
    assert metrics["noise_count"] == 0
    assert "silhouette_score" in metrics


def test_run_pipeline(sample_csv_path, tmp_path):
    output_path = str(tmp_path / "processed_output.csv")
    df_result, metrics = run_pipeline(sample_csv_path, output_path)

    assert os.path.exists(output_path)
    assert "cluster" in df_result.columns
    assert "cluster_keywords" in df_result.columns
    assert "n_clusters" in metrics


def test_llm_labeler_fallback():
    label_info = generate_fallback_label(0, ["credit", "card", "billing"])
    assert "theme_title" in label_info
    assert "summary" in label_info
    assert "Credit / Card / Billing Issues" in label_info["theme_title"]


def test_llm_labeler_unclustered():
    label_info = generate_llm_theme_label(-1, [], [])
    assert label_info["theme_title"] == "Unclustered / Noise Complaints"


def test_auto_label_all_clusters():
    keywords_map = {
        0: ["credit", "card"],
        1: ["mortgage", "escrow"]
    }
    labels_map = auto_label_all_clusters(keywords_map)
    assert 0 in labels_map
    assert 1 in labels_map
    assert "theme_title" in labels_map[0]

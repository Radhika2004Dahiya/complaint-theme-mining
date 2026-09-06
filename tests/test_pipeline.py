import unittest
import numpy as np
import pandas as pd
import os

from src.utils import clean_narrative, compute_cosine_distances
from src.data_loader import load_and_preprocess_complaints
from src.nlp_pipeline import evaluate_clustering
from src.llm_labeler import generate_fallback_label

class TestComplaintThemeMining(unittest.TestCase):

    def test_clean_narrative(self):
        text = "I paid on XX/XX/2021 to account XXXX with $50.00   extra fee."
        cleaned = clean_narrative(text)
        self.assertNotIn("XXXX", cleaned)
        self.assertNotIn("XX/XX/2021", cleaned)
        self.assertIn("[REDACTED]", cleaned)
        self.assertIn("[DATE]", cleaned)

    def test_cosine_distance(self):
        v = np.array([1.0, 0.0])
        m = np.array([[1.0, 0.0], [0.0, 1.0]])
        dists = compute_cosine_distances(v, m)
        self.assertAlmostEqual(dists[0], 0.0, places=5)
        self.assertAlmostEqual(dists[1], 1.0, places=5)

    def test_data_loader(self):
        df = load_and_preprocess_complaints("data/sample_data.csv", product_filter="Credit card", min_narrative_len=10)
        self.assertFalse(df.empty)
        self.assertIn("cleaned_narrative", df.columns)

    def test_clustering_eval(self):
        embeddings = np.random.randn(50, 10)
        labels = np.array([0]*20 + [1]*20 + [-1]*10)
        eval_res = evaluate_clustering(embeddings, labels)
        self.assertEqual(eval_res["n_total"], 50)
        self.assertEqual(eval_res["n_clusters"], 2)
        self.assertEqual(eval_res["noise_count"], 10)
        self.assertIsNotNone(eval_res["silhouette_score"])

if __name__ == "__main__":
    unittest.main()

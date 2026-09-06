"""
Utility functions for text cleaning, PII redaction, and metric computation.
"""

import re
import numpy as np
from typing import List, Union

def clean_narrative(text: Union[str, float]) -> str:
    """
    Clean and normalize complaint narrative text.
    - Replaces CFPB PII redaction placeholders (XXXX, XX/XX/XXXX) with standardized tokens.
    - Normalizes multi-whitespace characters.
    - Trims leading/trailing whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Replace date PII placeholders like XX/XX/XXXX or XX/XX/2021
    cleaned = re.sub(r'\bXX/XX/X+\b|\bXX/XX/\d{4}\b', '[DATE]', text, flags=re.IGNORECASE)

    # Replace general CFPB PII placeholders like XXXX, XXXXX
    cleaned = re.sub(r'\bX{2,}\b', '[REDACTED]', cleaned)

    # Replace multiple spaces / newlines with single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def compute_cosine_distances(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine distances between a 1D vector and a 2D matrix of vectors.
    Cosine distance = 1 - Cosine Similarity.
    """
    norm_v = np.linalg.norm(vector)
    norm_m = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero
    norm_v = max(norm_v, 1e-10)
    norm_m = np.maximum(norm_m, 1e-10)

    similarities = np.dot(matrix, vector) / (norm_m * norm_v)
    return 1.0 - similarities

"""
Similarity metrics for comparing embedding vectors.

This module provides various metrics for measuring similarity
between embedding vectors, useful for analyzing the shared
embedding space of Voyage AI models.
"""

import numpy as np
from typing import Union


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Cosine similarity score in range [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean (L2) distance between two vectors.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Euclidean distance (0 = identical, higher = more different)
    """
    return float(np.linalg.norm(a - b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute dot product between two vectors.

    Note: This is most meaningful when vectors are normalized.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Dot product score
    """
    return float(np.dot(a, b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Manhattan (L1) distance between two vectors.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Manhattan distance (0 = identical, higher = more different)
    """
    return float(np.sum(np.abs(a - b)))


def pairwise_similarities(
    embeddings: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a set of embeddings.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        metric: Similarity metric to use ("cosine", "euclidean", "dot", "manhattan")

    Returns:
        numpy array of shape (n_samples, n_samples) with pairwise similarities
    """
    n = len(embeddings)
    similarities = np.zeros((n, n))

    metric_funcs = {
        "cosine": cosine_similarity,
        "euclidean": euclidean_distance,
        "dot": dot_product,
        "manhattan": manhattan_distance
    }

    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_funcs.keys())}")

    func = metric_funcs[metric]

    for i in range(n):
        for j in range(n):
            similarities[i, j] = func(embeddings[i], embeddings[j])

    return similarities


def similarity_to_distance(similarity: float, metric: str = "cosine") -> float:
    """
    Convert a similarity score to a distance measure.

    Args:
        similarity: Similarity score
        metric: The metric used ("cosine" or "dot")

    Returns:
        Distance measure (0 = identical, higher = more different)
    """
    if metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        return 1.0 - similarity
    elif metric == "dot":
        # For normalized vectors, same as cosine
        return 1.0 - similarity
    else:
        raise ValueError(f"Conversion not supported for metric: {metric}")

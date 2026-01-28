"""
Diversity metrics for embedding population analysis.

This module provides metrics to measure how diverse or homogeneous
a population of embeddings is in the semantic space.
"""

import numpy as np
from typing import Optional
from .metrics import pairwise_similarities


def vendi_score(embeddings: np.ndarray, similarity_matrix: Optional[np.ndarray] = None) -> float:
    """
    Compute the Vendi Score for a set of embeddings.
    
    The Vendi Score (Friedman & Dieng, 2022) measures diversity as the
    exponential of the Shannon entropy of the eigenvalues of the similarity
    matrix. It can be interpreted as the "effective number of unique items."
    
    Higher score = more diversity.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        similarity_matrix: Optional precomputed similarity matrix. If None,
                          will compute cosine similarity.
    
    Returns:
        Vendi Score (float >= 1). A score of 1 means all items are identical.
        A score of n means n perfectly distinct items.
    """
    n = len(embeddings)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    
    # Compute similarity matrix if not provided
    if similarity_matrix is None:
        similarity_matrix = pairwise_similarities(embeddings, metric="cosine")
    
    # Normalize to make it a proper kernel matrix (divide by n)
    K = similarity_matrix / n
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(K)
    
    # Filter out near-zero or negative eigenvalues (numerical stability)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Compute Shannon entropy: -sum(λ * log(λ))
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    
    # Vendi Score is exp(entropy)
    return float(np.exp(entropy))


def mean_pairwise_similarity(
    embeddings: np.ndarray,
    similarity_matrix: Optional[np.ndarray] = None
) -> float:
    """
    Compute the mean pairwise cosine similarity (excluding self-similarity).
    
    Lower score = more diversity.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        similarity_matrix: Optional precomputed similarity matrix.
    
    Returns:
        Mean pairwise similarity in range [0, 1] for cosine similarity.
    """
    n = len(embeddings)
    if n < 2:
        return 1.0
    
    if similarity_matrix is None:
        similarity_matrix = pairwise_similarities(embeddings, metric="cosine")
    
    # Extract upper triangle (excluding diagonal)
    upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
    
    return float(np.mean(upper_tri))


def similarity_spread(
    embeddings: np.ndarray,
    similarity_matrix: Optional[np.ndarray] = None
) -> dict:
    """
    Compute statistics about the spread of pairwise similarities.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        similarity_matrix: Optional precomputed similarity matrix.
    
    Returns:
        Dictionary with 'std', 'iqr', 'min', 'max', 'range' statistics.
    """
    n = len(embeddings)
    if n < 2:
        return {"std": 0.0, "iqr": 0.0, "min": 1.0, "max": 1.0, "range": 0.0}
    
    if similarity_matrix is None:
        similarity_matrix = pairwise_similarities(embeddings, metric="cosine")
    
    # Extract upper triangle (excluding diagonal)
    upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
    
    q75, q25 = np.percentile(upper_tri, [75, 25])
    
    return {
        "std": float(np.std(upper_tri)),
        "iqr": float(q75 - q25),
        "min": float(np.min(upper_tri)),
        "max": float(np.max(upper_tri)),
        "range": float(np.max(upper_tri) - np.min(upper_tri))
    }


def effective_rank(embeddings: np.ndarray) -> float:
    """
    Compute the effective rank of the embedding matrix.
    
    Based on the Shannon entropy of normalized singular values.
    Measures how many dimensions effectively capture the variance.
    
    Higher score = more diverse directions used in embedding space.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
    
    Returns:
        Effective rank (float >= 1).
    """
    n = len(embeddings)
    if n < 2:
        return 1.0
    
    # Center the embeddings
    centered = embeddings - np.mean(embeddings, axis=0)
    
    # Compute singular values
    singular_values = np.linalg.svd(centered, compute_uv=False)
    
    # Normalize to get a probability distribution
    singular_values = singular_values[singular_values > 1e-10]
    p = singular_values / np.sum(singular_values)
    
    # Compute entropy
    entropy = -np.sum(p * np.log(p))
    
    # Effective rank is exp(entropy)
    return float(np.exp(entropy))


def diversity_report(embeddings: np.ndarray) -> dict:
    """
    Generate a complete diversity report for a set of embeddings.
    
    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
    
    Returns:
        Dictionary containing all diversity metrics.
    """
    # Compute similarity matrix once for reuse
    sim_matrix = pairwise_similarities(embeddings, metric="cosine")
    
    spread = similarity_spread(embeddings, sim_matrix)
    
    return {
        "n_samples": len(embeddings),
        "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        "vendi_score": vendi_score(embeddings, sim_matrix),
        "mean_pairwise_similarity": mean_pairwise_similarity(embeddings, sim_matrix),
        "effective_rank": effective_rank(embeddings),
        "similarity_std": spread["std"],
        "similarity_iqr": spread["iqr"],
        "similarity_range": spread["range"],
        "similarity_min": spread["min"],
        "similarity_max": spread["max"],
    }

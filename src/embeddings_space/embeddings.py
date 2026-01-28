"""
Voyage AI embeddings client wrapper.

This module provides a simple interface for generating embeddings
using Voyage AI's models, particularly the Voyage 4 series which
share a common embedding space.
"""

import os
from typing import Optional

import numpy as np
import voyageai
from dotenv import load_dotenv


def load_api_key() -> str:
    """Load Voyage AI API key from environment."""
    load_dotenv()
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "VOYAGE_API_KEY not found. Please set it in your .env file."
        )
    return api_key


class EmbeddingsClient:
    """Client for generating embeddings using Voyage AI."""

    # Voyage 4 models that share a common embedding space
    VOYAGE_4_MODELS = [
        "voyage-4-large",       # Newest, most capable model
        "voyage-4-lite",        # Lighter/faster version
        "voyage-3.5",           # General-purpose model
        "voyage-3.5-lite",      # Lighter/faster version
        "voyage-code-3",        # Optimized for code
        "voyage-finance-2",     # Optimized for finance
        "voyage-law-2",         # Optimized for legal
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-4-large"
    ):
        """
        Initialize the embeddings client.

        Args:
            api_key: Voyage AI API key. If not provided, loads from environment.
            model: Model to use for embeddings. Defaults to voyage-3.5.
        """
        self.api_key = api_key or load_api_key()
        self.model = model
        self.client = voyageai.Client(api_key=self.api_key)

    def embed_texts(
        self,
        texts: list[str],
        input_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            input_type: Optional input type hint ("query" or "document").

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type
        )
        return np.array(result.embeddings)

    def embed_single(
        self,
        text: str,
        input_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed.
            input_type: Optional input type hint ("query" or "document").

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.embed_texts([text], input_type=input_type)
        return embeddings[0]

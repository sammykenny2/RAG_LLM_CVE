"""
Embedding model management module.
Provides unified interface for SentenceTransformer embedding generation.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Import configuration
from config import (
    EMBEDDING_MODEL_NAME,
    CUDA_DEVICE,
    VERBOSE_LOGGING,
    RETRIEVAL_TOP_K
)


class EmbeddingModel:
    """
    Wrapper for SentenceTransformer embedding model.

    Example usage:
        embedder = EmbeddingModel()
        embedder.initialize()
        embeddings = embedder.encode(["text1", "text2"])
        embedder.cleanup()
    """

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize embedding model wrapper.

        Args:
            model_name: Hugging Face model ID (default from config.py)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.device = device or self._auto_detect_device()
        self.model = None
        self._initialized = False

    def _auto_detect_device(self) -> str:
        """Automatically detect best available device."""
        if CUDA_DEVICE is not None:
            if CUDA_DEVICE == -1:
                return "cpu"
            else:
                return f"cuda:{CUDA_DEVICE}"

        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def initialize(self):
        """Load embedding model."""
        if self._initialized:
            if VERBOSE_LOGGING:
                print("[WARNING] Embedding model already initialized, skipping...")
            return self.model

        if VERBOSE_LOGGING:
            print(f"Loading embedding model: {self.model_name}")
            print(f"  └─ Device: {self.device}")

        self.model = SentenceTransformer(
            model_name_or_path=self.model_name,
            device=self.device,
            local_files_only=True  # Use local cache only, no network required
        )

        self._initialized = True

        if VERBOSE_LOGGING:
            print(f"[OK] Embedding model loaded on {self.device}")

        return self.model

    def get_model(self):
        """
        Get embedding model (initializes if not already loaded).

        Returns:
            SentenceTransformer: Embedding model
        """
        if not self._initialized:
            self.initialize()
        return self.model

    def encode(
        self,
        texts,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        precision: str = None
    ):
        """
        Encode texts to embeddings.

        Args:
            texts: String or list of strings to encode
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar during encoding
            convert_to_numpy: Convert to numpy array
            convert_to_tensor: Convert to torch tensor
            precision: 'float32' or 'float16' (optional post-processing)

        Returns:
            Embeddings as numpy array or torch tensor
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() or get_model() first.")

        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor
        )

        # Apply precision conversion if requested
        if precision == 'float16':
            if convert_to_numpy:
                embeddings = embeddings.astype(np.float16)
            elif convert_to_tensor:
                embeddings = embeddings.to(torch.float16)
            if VERBOSE_LOGGING:
                print(f"  └─ Converted embeddings to float16")

        return embeddings

    def retrieve_top_k(
        self,
        query: str,
        embeddings: torch.Tensor,
        top_k: int = RETRIEVAL_TOP_K
    ) -> tuple:
        """
        Retrieve top-k most similar embeddings using dot product.

        Args:
            query: Query string to embed
            embeddings: Tensor of precomputed embeddings (N x D)
            top_k: Number of results to return

        Returns:
            tuple: (scores, indices) as torch tensors
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() or get_model() first.")

        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        # Ensure dtype consistency
        if query_embedding.dtype != embeddings.dtype:
            query_embedding = query_embedding.to(embeddings.dtype)

        # Compute dot product scores
        dot_scores = util.dot_score(query_embedding, embeddings)[0]

        # Get top-k
        scores, indices = torch.topk(input=dot_scores, k=top_k)

        return scores, indices

    def cleanup(self):
        """Clean up model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

        torch.cuda.empty_cache()

        self._initialized = False

        if VERBOSE_LOGGING:
            print("[OK] Embedding model cleaned up from memory")


# =============================================================================
# Utility functions (backward compatible)
# =============================================================================

def initialize_embedding_model(device=None):
    """
    Initialize embedding model (backward compatible function).

    Args:
        device: Device to use (None for auto-detect)

    Returns:
        SentenceTransformer: Embedding model
    """
    embedder = EmbeddingModel(device=device)
    return embedder.initialize()


def retrieve_context(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer,
    n_resources_to_return: int = 5
):
    """
    Retrieve top-k indices (backward compatible function).

    Args:
        query: Query string
        embeddings: Precomputed embeddings
        model: SentenceTransformer model
        n_resources_to_return: Number of results

    Returns:
        torch.Tensor: Indices of top-k results
    """
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Ensure dtype consistency
    if query_embedding.dtype != embeddings.dtype:
        query_embedding = query_embedding.to(embeddings.dtype)

    # Compute scores
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    # Get top-k
    _, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return indices

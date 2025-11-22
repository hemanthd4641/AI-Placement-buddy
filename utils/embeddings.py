"""Embedding adapter with configurable provider.

This module exposes `embed_texts(texts, dim=768)` which returns a numpy array of
embeddings (dtype float32). By default it prefers a local `sentence-transformers`
model (`all-MiniLM-L6-v2`) to avoid using paid/quotad Google embedding APIs.

You can override the provider with the environment variable `EMBEDDING_PROVIDER`:
- `local` (default): prefer `sentence-transformers` then Gemini then hash.
- `gemini`: prefer Google Gemini embeddings then local then hash.
- `auto`: try local then gemini then hash.
"""
from typing import List
import os
import numpy as np

try:
    from utils.model_manager import ModelManager
    MM_AVAILABLE = True
except Exception:
    ModelManager = None
    MM_AVAILABLE = False

# Try to import sentence-transformers for a local provider
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    ST_AVAILABLE = False

# Keep a module-level embedder instance for SentenceTransformer to avoid reloads
_st_embedder = None


def _hash_vector(text: str, dim: int = 768):
    # Simple deterministic fallback: repeated hashed values
    h = abs(hash(text)) % (10 ** 8)
    v = np.array([((h >> (i % 32)) & 0xFF) / 255.0 for i in range(dim)], dtype="float32")
    # Normalize
    try:
        import faiss

        faiss.normalize_L2(v)
    except Exception:
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
    return v


def _normalize_embeddings(embs: np.ndarray):
    try:
        import faiss

        faiss.normalize_L2(embs)
    except Exception:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
    return embs


def embed_texts(texts, dim: int = 768):
    """Return embeddings for `texts`.

    Behavior is controlled by `EMBEDDING_PROVIDER` env var. See module docstring.
    """
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()

    # Define provider order based on requested provider
    if provider == "gemini":
        order = ["gemini", "local"]
    else:
        # default and 'auto' both prefer local first
        order = ["local", "gemini"]

    # Attempt providers in order
    for p in order:
        if p == "local" and ST_AVAILABLE:
            try:
                global _st_embedder
                if _st_embedder is None:
                    _st_embedder = SentenceTransformer("all-MiniLM-L6-v2")
                embs = _st_embedder.encode(texts, show_progress_bar=False)
                embs = np.array(embs, dtype="float32")
                if embs.ndim == 1:
                    embs = embs.reshape(1, -1)
                embs = _normalize_embeddings(embs)
                return embs[0] if single else embs
            except Exception:
                # try next provider
                pass

        if p == "gemini" and MM_AVAILABLE:
            try:
                mm = ModelManager()
                embs = mm.get_embeddings(texts)
                embs = np.array(embs, dtype="float32")
                if embs.ndim == 1:
                    embs = embs.reshape(1, -1)
                embs = _normalize_embeddings(embs)
                return embs[0] if single else embs
            except Exception:
                pass

    # Deterministic fallback
    out = np.stack([_hash_vector(t, dim=dim) for t in texts])
    return out[0] if single else out

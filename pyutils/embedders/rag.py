"""
rag.py — RAG pipeline with FAISS retrieval and late chunking.

Components:
    Chunk        — a piece of text with its source metadata
    FAISSStore   — exact cosine search via IndexFlatIP on L2-normed vectors
    RAGPipeline  — index documents (late chunking) → query → return top-k

Requirements:
    pip install transformers>=4.51.0 torch numpy faiss-cpu

Quick-start:
    from rag import RAGPipeline

    rag = RAGPipeline()
    rag.index(["Your long document...", "Another document..."])
    results = rag.query("What is X?", top_k=5)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:80]}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import faiss
import numpy as np

from .embedder import QwenEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_idx: int
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FAISS Store
# ---------------------------------------------------------------------------

class FAISSStore:
    """
    Exact cosine similarity search using FAISS IndexFlatIP.

    Because all vectors are L2-normalised before insertion,
    inner product = cosine similarity.  IndexFlatIP is brute-force
    (no approximation, no quantisation) — accuracy is exact.

    For >1M vectors, swap to IndexIVFFlat for speed at the cost
    of a small recall penalty.  This store prioritises correctness.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks:     list of Chunk (length N)
            embeddings: (N, dim) float32, must be L2-normalised
        """
        if embeddings.shape != (len(chunks), self.dim):
            raise ValueError(
                f"Shape mismatch: {embeddings.shape} vs expected ({len(chunks)}, {self.dim})"
            )
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Find the top_k chunks most similar to query_vec.

        Args:
            query_vec: (dim,) float32, L2-normalised
            top_k:     number of results

        Returns:
            List of dicts, descending by cosine score:
                score     : float in [-1, 1]
                text      : chunk text
                doc_id    : source document id
                chunk_idx : position within document
                metadata  : passthrough dict
        """
        if self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(
            query_vec.reshape(1, -1), top_k,
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                               # FAISS sentinel
                continue
            c = self._chunks[idx]
            results.append({
                "score": float(score),
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_idx": c.chunk_idx,
                "metadata": c.metadata,
            })
        return results

    def __len__(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class RAGPipeline:
    """
    End-to-end retrieval pipeline.

    Indexing uses late chunking by default: each document gets a single
    forward pass through the transformer, then the hidden states are split
    into chunk-sized windows and mean-pooled.  Cross-chunk context is
    preserved because every token's hidden state already carries information
    from all preceding tokens.

    Querying uses standard last-token-pool encoding with an instruction
    prefix for best retrieval accuracy.

    Usage:
        rag = RAGPipeline()
        rag.index(["doc text 1", "doc text 2"], doc_ids=["d1", "d2"])
        results = rag.query("your question", top_k=5)
    """

    def __init__(
        self,
        embedder: QwenEmbedder | None = None,
        chunk_tokens: int = 512,
        task: str = DEFAULT_TASK,
    ) -> None:
        """
        Args:
            embedder:     QwenEmbedder instance (created with defaults if None)
            chunk_tokens: tokens per late-chunking window
            task:         instruction prepended to queries
        """
        self.embedder = embedder or QwenEmbedder()
        self.chunk_tokens = chunk_tokens
        self.task = task
        self.store = FAISSStore(dim=self.embedder.dim)

    def index(
        self,
        documents: List[str],
        doc_ids: List[str] | None = None,
        metadatas: List[Dict] | None = None,
    ) -> None:
        """
        Index documents using late chunking.

        Each document → one forward pass → multiple contextual chunk embeddings.

        Args:
            documents: raw text strings (any length)
            doc_ids:   stable IDs; defaults to "doc_0", "doc_1", ...
            metadatas: per-document metadata dicts
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        total_chunks = 0

        for text, doc_id, meta in zip(documents, doc_ids, metadatas):
            if not text.strip():
                continue

            chunk_texts, embeddings = self.embedder.encode_document(
                text, chunk_tokens=self.chunk_tokens,
            )

            chunks = [
                Chunk(text=ct, doc_id=doc_id, chunk_idx=j, metadata=meta)
                for j, ct in enumerate(chunk_texts)
            ]

            self.store.add(chunks, embeddings)
            total_chunks += len(chunks)

        logger.info(
            "Indexed %d chunks from %d documents.", total_chunks, len(documents),
        )

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        """
        Embed query and return top-k most relevant chunks.

        The query is encoded with an instruction prefix using last-token
        pooling (standard mode), matching how the model was trained.

        Returns:
            List of result dicts, each with:
                score, text, doc_id, chunk_idx, metadata
        """
        query_vec = self.embedder.encode(
            text, is_query=True, task=self.task,
        )[0]
        return self.store.search(query_vec, top_k=top_k)

    def __len__(self) -> int:
        return len(self.store)
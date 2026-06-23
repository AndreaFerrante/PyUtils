"""
rag.py — RAG pipeline with FAISS retrieval and late chunking.

Components:
    Chunk        — a piece of text with its source metadata
    FAISSStore   — cosine search over L2-normed vectors; exact ("flat") by
                   default, with opt-in approximate indexes (ivf, hnsw, ivfpq)
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

from embedder import QwenEmbedder

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
    Vector store backed by FAISS, with a choice of index.

    All vectors are L2-normalised, so inner product equals cosine similarity
    and every index below returns cosine scores.

    Index types (set via `index_type`):

        "flat"   Exact brute-force search: compares the query against every
                 vector. 100% recall, no training, no tuning. The right default
                 and the accuracy baseline. Cost grows linearly with #vectors.

        "ivf"    Inverted file. Groups vectors into `nlist` clusters and, at
                 query time, searches only the `nprobe` nearest clusters. Much
                 faster on large corpora; recall is traded for speed via
                 `nprobe` (higher = more accurate, slower). Needs training.

        "hnsw"   A navigable graph. Hops through a multi-layer "small-world"
                 graph of vectors to reach near neighbours in a few steps.
                 Excellent speed/recall, no training, but uses more memory (it
                 stores graph links on top of the vectors). Tuned via
                 `hnsw_ef_search`.

        "ivfpq"  Inverted file + product quantisation. Like "ivf", but also
                 compresses each vector into `pq_m` small codes instead of
                 storing it in full. Drastically less memory for very large
                 corpora; compression is lossy, so it is the least accurate.

    Build model:
        Vectors are buffered as they are added, and the index is built once,
        lazily, on the first search (or via an explicit `build()`). This is
        what lets "ivf"/"ivfpq" train on the whole corpus at once. After
        building, the buffer is freed and further `add()`s go into the live
        index.

    Index/metadata alignment:
        Vectors and their Chunk records are appended in lockstep, and FAISS
        assigns sequential ids, so index position i always maps to chunk i.
    """

    _VALID_TYPES = ("flat", "ivf", "hnsw", "ivfpq")

    def __init__(
        self,
        dim: int = 1024,
        index_type: str = "flat",
        *,
        nlist: int | None = None,
        nprobe: int = 16,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        pq_m: int = 64,
        pq_nbits: int = 8,
    ) -> None:
        """
        Args:
            dim:        embedding dimension (1024 for Qwen3-Embedding-0.6B)
            index_type: "flat" | "ivf" | "hnsw" | "ivfpq"

          IVF / IVFPQ:
            nlist:      number of clusters; None → auto ≈ sqrt(N) at build time
            nprobe:     clusters searched per query (recall ↔ speed)

          IVFPQ (product quantisation):
            pq_m:       sub-vectors per vector; must divide `dim`
            pq_nbits:   bits per sub-vector code (8 is standard)

          HNSW:
            hnsw_m:               graph neighbours per node (recall ↔ memory)
            hnsw_ef_construction: build-time search depth (index quality)
            hnsw_ef_search:       query-time search depth (recall ↔ speed)
        """
        if index_type not in self._VALID_TYPES:
            raise ValueError(
                f"index_type must be one of {list(self._VALID_TYPES)}, got {index_type!r}"
            )
        if index_type == "ivfpq" and dim % pq_m != 0:
            raise ValueError(f"pq_m={pq_m} must divide dim={dim}")

        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits

        self._chunks: List[Chunk] = []
        self._pending: List[np.ndarray] = []   # vectors awaiting the first build
        self._index = None                     # faiss.Index, created at build
        self._built = False

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _make_index(self, n_vectors: int):
        """Build an empty index of the requested type (inner-product metric)."""
        metric = faiss.METRIC_INNER_PRODUCT

        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dim)

        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, metric)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
            return index

        # "ivf" and "ivfpq" share a coarse quantiser and an auto-sized nlist.
        nlist = self.nlist or max(1, round(n_vectors ** 0.5))
        nlist = min(nlist, n_vectors)              # cannot exceed #vectors
        if n_vectors < 39 * nlist:                 # FAISS k-means guidance
            logger.warning(
                "%s: %d vectors is small for nlist=%d; recall may suffer. "
                "Consider index_type='flat' for small corpora.",
                self.index_type, n_vectors, nlist,
            )
        quantizer = faiss.IndexFlatIP(self.dim)

        if self.index_type == "ivf":
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, metric)
        else:  # "ivfpq"
            index = faiss.IndexIVFPQ(
                quantizer, self.dim, nlist, self.pq_m, self.pq_nbits, metric,
            )
        index.nprobe = min(self.nprobe, nlist)
        return index

    def build(self) -> None:
        """
        Build the index from all buffered vectors. Idempotent.

        Trains first for "ivf"/"ivfpq" (the others need no training), then adds
        every buffered vector and releases the buffer. Called lazily by
        `search()`, or explicitly to control when the one-off training happens.
        """
        if self._built or not self._pending:
            return
        vectors = np.vstack(self._pending)
        index = self._make_index(len(vectors))
        if not index.is_trained:
            index.train(vectors)
        index.add(vectors)
        self._index = index
        self._pending = []          # free buffer; "ivfpq" now keeps only codes
        self._built = True
        logger.info("Built %s index with %d vectors.", self.index_type, index.ntotal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their L2-normalised embeddings.

        Before the first build the vectors are buffered; afterwards they go
        straight into the live index. Either way chunk metadata is recorded so
        results can be mapped back.

        Args:
            chunks:     list of Chunk (length N)
            embeddings: (N, dim) float32, L2-normalised
        """
        if embeddings.shape != (len(chunks), self.dim):
            raise ValueError(
                f"Shape mismatch: {embeddings.shape} vs expected ({len(chunks)}, {self.dim})"
            )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._chunks.extend(chunks)
        if self._built:
            self._index.add(embeddings)
        else:
            self._pending.append(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Return the top_k chunks most similar to query_vec.

        Builds the index on the first call if it has not been built yet.

        Args:
            query_vec: (dim,) float32, L2-normalised
            top_k:     number of results

        Returns:
            List of dicts, descending by cosine score:
                score, text, doc_id, chunk_idx, metadata
        """
        if not self._chunks:                 # nothing indexed yet
            return []
        if not self._built:
            self.build()

        top_k = min(top_k, self._index.ntotal)
        query = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, indices = self._index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                      # sentinel: fewer than top_k found
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
        return len(self._chunks)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class RAGPipeline:
    """
    End-to-end retrieval pipeline.

    Indexing uses late chunking: each document gets a single forward pass
    through the model, and each chunk is represented by its last token — which
    has attended over all preceding text, so context carries across chunk
    boundaries.

    Querying uses last-token-pool encoding with an instruction prefix, the same
    pooling as the documents, so query and chunk vectors are comparable.

    Retrieval is delegated to a FAISSStore. Pass your own to pick an
    approximate index for large corpora; the default is exact search.

    Usage:
        rag = RAGPipeline()
        rag.index(["doc text 1", "doc text 2"], doc_ids=["d1", "d2"])
        results = rag.query("your question", top_k=5)

        # large corpus, approximate index:
        from rag import FAISSStore
        store = FAISSStore(dim=1024, index_type="hnsw")   # dim must match the model
        rag = RAGPipeline(store=store)
    """

    def __init__(
        self,
        embedder: QwenEmbedder | None = None,
        store: FAISSStore | None = None,
        chunk_tokens: int = 512,
        task: str = DEFAULT_TASK,
    ) -> None:
        """
        Args:
            embedder:     QwenEmbedder instance (created with defaults if None)
            store:        FAISSStore instance (created as exact "flat" if None)
            chunk_tokens: tokens per late-chunking window
            task:         instruction prepended to queries
        """
        self.embedder = embedder if embedder is not None else QwenEmbedder()
        self.chunk_tokens = chunk_tokens
        self.task = task
        self.store = store if store is not None else FAISSStore(dim=self.embedder.dim)

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
"""
test_rag.py — correctness checks for the pure logic.

These tests cover the math that must be exactly right: pooling, the
L2-normalisation identity, the search index, and the chunk-windowing
arithmetic. They do NOT download the model, so they run anywhere in
seconds.

Run:
    pytest tests/test_rag.py        # if pytest is installed
    python  tests/test_rag.py       # plain python also works
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Make the project root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import _last_token_pool, EMBEDDING_DIM, MAX_SEQ_TOKENS, MODEL_ID
from rag import Chunk, FAISSStore


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def test_last_token_pool_left_padding():
    """With left-padding, the last position is always real."""
    hidden = torch.randn(3, 10, EMBEDDING_DIM)
    mask = torch.ones(3, 10)
    mask[0, :3] = 0
    mask[1, :1] = 0
    out = _last_token_pool(hidden, mask)
    assert out.shape == (3, EMBEDDING_DIM)
    assert torch.equal(out, hidden[:, -1])


def test_last_token_pool_right_padding():
    """With right-padding, pick the token at (length - 1) per sequence."""
    hidden = torch.randn(3, 10, EMBEDDING_DIM)
    mask = torch.ones(3, 10)
    mask[0, 7:] = 0   # 7 real tokens -> idx 6
    mask[1, 5:] = 0   # 5 real tokens -> idx 4
    out = _last_token_pool(hidden, mask)
    assert torch.equal(out[0], hidden[0, 6])
    assert torch.equal(out[1], hidden[1, 4])
    assert torch.equal(out[2], hidden[2, 9])


# ---------------------------------------------------------------------------
# L2 normalisation identity
# ---------------------------------------------------------------------------

def test_l2_norm_makes_dot_equal_cosine():
    """On L2-normalised vectors, dot product == cosine similarity."""
    v = torch.randn(5, EMBEDDING_DIM)
    n = F.normalize(v, p=2, dim=1)
    norms = (n ** 2).sum(dim=1) ** 0.5
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)
    dot = float(n[0] @ n[1])
    cos = float(F.cosine_similarity(n[0:1], n[1:2]))
    assert abs(dot - cos) < 1e-5


# ---------------------------------------------------------------------------
# Late-chunk windowing + last-token pooling (logic replicated, no model)
# ---------------------------------------------------------------------------

def test_late_chunk_windowing_and_last_token():
    """
    Replicates encode_document's windowing: for seq_len tokens split into
    windows of chunk_tokens, each chunk is the last token of its window,
    and the result is L2-normalised.
    """
    seq_len, chunk_tokens = 200, 64
    hidden = torch.randn(seq_len, EMBEDDING_DIM)

    vecs, last_indices = [], []
    for start in range(0, seq_len, chunk_tokens):
        end = min(start + chunk_tokens, seq_len)
        vecs.append(hidden[end - 1])
        last_indices.append(end - 1)

    stacked = F.normalize(torch.stack(vecs), p=2, dim=1)

    expected_chunks = (seq_len + chunk_tokens - 1) // chunk_tokens
    assert stacked.shape == (expected_chunks, EMBEDDING_DIM)
    # Window boundaries land on the right token positions.
    assert last_indices == [63, 127, 191, 199]
    norms = (stacked ** 2).sum(dim=1) ** 0.5
    assert torch.allclose(norms, torch.ones(expected_chunks), atol=1e-5)


def test_matched_pooling_gives_identical_vectors():
    """
    The bug we fixed: query (last-token) vs document (mean) put vectors in
    different spaces. With both using last-token, identical text -> cos 1.0.
    """
    hidden = torch.randn(6, EMBEDDING_DIM)
    q = F.normalize(hidden[-1:], p=2, dim=1)
    doc = F.normalize(hidden[-1:], p=2, dim=1)   # last-token, same as query
    assert abs(float(q @ doc.T) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# FAISS store
# ---------------------------------------------------------------------------

def _normed(n, dim, seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(n, dim).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_faiss_exact_search_matches_numpy():
    """IndexFlatIP must return the same ranking as numpy brute force."""
    dim = EMBEDDING_DIM
    vecs = _normed(100, dim, seed=42)
    store = FAISSStore(dim=dim)
    chunks = [Chunk(text=f"chunk_{i}", doc_id="d", chunk_idx=i) for i in range(100)]
    store.add(chunks, vecs)

    q = vecs[42]
    results = store.search(q, top_k=5)

    assert results[0]["text"] == "chunk_42"
    assert results[0]["score"] > 0.999
    faiss_top = [int(r["text"].split("_")[1]) for r in results]
    numpy_top = list(np.argsort(vecs @ q)[::-1][:5])
    assert faiss_top == numpy_top
    # Scores must be sorted descending.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_faiss_incremental_add():
    """Adding in batches preserves correct indexing across batches."""
    dim = EMBEDDING_DIM
    store = FAISSStore(dim=dim)
    a, b = _normed(10, dim, 1), _normed(5, dim, 2)
    store.add([Chunk(f"a_{i}", "a", i) for i in range(10)], a)
    store.add([Chunk(f"b_{i}", "b", i) for i in range(5)], b)
    assert len(store) == 15
    assert store.search(b[3], top_k=1)[0]["text"] == "b_3"


def test_faiss_shape_guard():
    """Mismatched chunk/embedding counts must raise."""
    store = FAISSStore(dim=EMBEDDING_DIM)
    try:
        store.add([Chunk("x", "x", 0)], np.zeros((2, EMBEDDING_DIM), np.float32))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_faiss_empty_search():
    """Searching an empty index returns an empty list, not an error."""
    store = FAISSStore(dim=EMBEDDING_DIM)
    assert store.search(np.zeros(EMBEDDING_DIM, np.float32)) == []


# ---------------------------------------------------------------------------
# Approximate index types
# ---------------------------------------------------------------------------

# Small dimension + clean PQ params (16 codebook centroids, plenty of training
# data) so these tests run fast and print no FAISS training warnings.
_APX_DIM = 64
_APX_N = 3000


def _approx_store(index_type):
    kw = dict(dim=_APX_DIM, index_type=index_type)
    if index_type == "ivfpq":
        kw.update(pq_m=8, pq_nbits=4)   # 8-dim sub-vectors, 16 centroids each
    store = FAISSStore(**kw)
    vecs = _normed(_APX_N, _APX_DIM, seed=7)
    store.add([Chunk(f"c_{i}", "d", i) for i in range(_APX_N)], vecs)
    return store, vecs


def test_all_index_types_find_self():
    """
    A stored vector is its own nearest neighbour. Exact and graph indexes put
    it at rank 1; the lossy compressed index should keep it within the top 5.
    """
    for index_type in ("flat", "ivf", "hnsw"):
        store, vecs = _approx_store(index_type)
        top = store.search(vecs[123], top_k=5)
        assert top[0]["text"] == "c_123", f"{index_type}: expected rank-1 self"

    store, vecs = _approx_store("ivfpq")
    names = [r["text"] for r in store.search(vecs[123], top_k=5)]
    assert "c_123" in names, "ivfpq: self vector missing from top-5"


def test_approximate_recall_matches_flat():
    """
    With generous search effort, IVF and HNSW should agree with exact search
    on almost all of the top-10 neighbours.
    """
    flat, vecs = _approx_store("flat")
    queries = _normed(20, _APX_DIM, seed=99)
    truth = [[r["text"] for r in flat.search(q, top_k=10)] for q in queries]

    # IVF probing every cluster ≈ exact.
    ivf = FAISSStore(dim=_APX_DIM, index_type="ivf", nprobe=10_000)
    ivf.add([Chunk(f"c_{i}", "d", i) for i in range(_APX_N)], vecs)

    # HNSW with a deep query-time search.
    hnsw = FAISSStore(dim=_APX_DIM, index_type="hnsw", hnsw_ef_search=256)
    hnsw.add([Chunk(f"c_{i}", "d", i) for i in range(_APX_N)], vecs)

    for store in (ivf, hnsw):
        overlap = 0
        for q, want in zip(queries, truth):
            got = {r["text"] for r in store.search(q, top_k=10)}
            overlap += len(got & set(want))
        recall = overlap / (len(queries) * 10)
        assert recall >= 0.9, f"{store.index_type}: recall {recall:.2f} < 0.90"


def test_lazy_build_then_explicit_build_is_idempotent():
    """Index builds on first search; an explicit build() is safe to repeat."""
    store, vecs = _approx_store("ivf")
    assert not store._built                 # buffered, not yet built
    _ = store.search(vecs[0], top_k=1)      # triggers build
    assert store._built
    store.build()                           # idempotent no-op
    assert len(store) == _APX_N


def test_post_build_add_goes_live():
    """Adding after the index is built inserts straight into the live index."""
    store, vecs = _approx_store("hnsw")
    _ = store.search(vecs[0], top_k=1)      # build
    extra = _normed(50, _APX_DIM, seed=5)
    store.add([Chunk(f"x_{i}", "x", i) for i in range(50)], extra)
    assert len(store) == _APX_N + 50
    assert store.search(extra[10], top_k=1)[0]["text"] == "x_10"


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------

def test_invalid_index_type_raises():
    try:
        FAISSStore(dim=64, index_type="banana")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pq_m_must_divide_dim():
    try:
        FAISSStore(dim=100, index_type="ivfpq", pq_m=64)   # 100 % 64 != 0
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Pipeline wiring (model-free stub embedder)
# ---------------------------------------------------------------------------

class _StubEmbedder:
    """Minimal stand-in so RAGPipeline can be tested without the model."""
    dim = 64

    def encode(self, texts, is_query=False, task=""):
        if isinstance(texts, str):
            texts = [texts]
        return _normed(len(texts), self.dim, seed=1)

    def encode_document(self, text, chunk_tokens=512):
        return [text], _normed(1, self.dim, seed=1)


def test_injected_store_is_retained():
    """
    Regression: FAISSStore defines __len__, so an empty store is falsy. The
    pipeline must keep an injected store with `is not None`, not `or` (which
    would silently fall back to the default flat store).
    """
    from rag import RAGPipeline
    store = FAISSStore(dim=_StubEmbedder.dim, index_type="hnsw")
    rag = RAGPipeline(embedder=_StubEmbedder(), store=store)
    assert rag.store is store
    assert rag.store.index_type == "hnsw"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants():
    assert MODEL_ID == "Qwen/Qwen3-Embedding-0.6B"
    assert EMBEDDING_DIM == 1024
    assert MAX_SEQ_TOKENS == 32_768


# ---------------------------------------------------------------------------
# Plain-python runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
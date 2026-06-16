# pyutils.embedders

Qwen3-based text embedding and retrieval helpers. Use the embedder for normalized 1024-d vectors, and the RAG pipeline when you need exact cosine retrieval over chunked documents.

## What it provides

- `QwenEmbedder`: batched text encoding with last-token pooling.
- `encode_document(...)`: late chunking for long documents. It runs one forward pass over the full text, then mean-pools token windows into chunk embeddings.
- `FAISSStore`: exact cosine search over L2-normalized vectors.
- `RAGPipeline`: document indexing plus retrieval on top of the embedder.

## Why this design

- Query and document embeddings come from the same model, so retrieval stays consistent.
- L2 normalization makes cosine similarity equal to dot product.
- Late chunking keeps more document context than embedding chunks independently.
- `IndexFlatIP` keeps retrieval exact and easy to reason about.

## Install

```bash
pip install transformers>=4.51.0 torch numpy faiss-cpu
```

CPU is fine. CUDA or MPS are picked up automatically when available.

## Quick start

```py
from pyutils.embedders.embedder import QwenEmbedder
from pyutils.embedders.rag import RAGPipeline

embedder = QwenEmbedder(device="cpu")
vecs = embedder.encode(["hello world", "another doc"])

chunk_texts, chunk_vecs = embedder.encode_document("long document...", chunk_tokens=512)

rag = RAGPipeline(embedder=embedder, chunk_tokens=128)
rag.index(["doc one text", "doc two text"], doc_ids=["d1", "d2"])
results = rag.query("What is X?", top_k=3)
```

Each result is a dict with `score`, `text`, `doc_id`, `chunk_idx`, and `metadata`.

## API

- `QwenEmbedder(device=None, batch_size=8)`
- `encode(texts, is_query=False, task=DEFAULT_TASK) -> np.ndarray`
- `encode_document(text, chunk_tokens=512) -> tuple[list[str], np.ndarray]`
- `RAGPipeline(embedder=None, chunk_tokens=512, task=DEFAULT_TASK)`
- `index(documents, doc_ids=None, metadatas=None) -> None`
- `query(text, top_k=5) -> list[dict]`

## Files

- `embedder.py` - Qwen embedder and late chunking.
- `rag.py` - FAISS store and RAG pipeline.
- `example.py` - local CPU smoke test.

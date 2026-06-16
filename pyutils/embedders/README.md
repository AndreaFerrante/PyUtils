# pyutils.embedders

Lightweight text embedding and RAG utilities using the `all-MiniLM-L6-v2` model.

This package provides:
- `MiniLMEmbedder` — production-ready encoder producing L2-normalised 384-d vectors.
- `Chunker`, `VectorStore`, `RAGPipeline` — minimal RAG pipeline for indexing and retrieval.

## Install

Install the recommended stack:

```bash
pip install sentence-transformers>=5.0.0 numpy
```

You can avoid the `sentence-transformers` dependency by using the `transformers` backend, but you'll need `transformers` and `torch` installed.

## Quick usage

Minimal embedder usage:

```py
from pyutils.embedders import MiniLMEmbedder

embedder = MiniLMEmbedder()  # auto device, sentence-transformers backend
vecs = embedder.encode(["hello world", "another doc"])  # shape (N, 384), float32
```

Notes:
- Returned vectors are L2-normalised; cosine similarity == dot product.
- The model has a hard token limit of 256 word-pieces; chunk long documents first.

RAG quick-start:

```py
from pyutils.embedders.rag import RAGPipeline

rag = RAGPipeline()  # uses MiniLMEmbedder + default Chunker
rag.index(["Long document text...", "Another document..."], doc_ids=["d1","d2"])
results = rag.query("What is X?", top_k=3)
for r in results:
    print(r["score"], r["doc_id"], r["text"])
```

APIs (concise)
- `MiniLMEmbedder(device=None, batch_size=64, backend='sentence-transformers', cache_dir=None)`
  - `encode(texts: str|List[str], show_progress=False) -> np.ndarray` — returns (N,384) float32 L2-normalised vectors.
- `Chunker(max_tokens=200, overlap_tokens=20)` — token-precise splitter; use before embedding long docs.
- `RAGPipeline(embedder=None, chunker=None, show_progress=False)`
  - `index(documents: List[str], doc_ids: List[str]|None=None, metadatas: List[dict]|None=None)`
  - `query(query: str, top_k: int = 5) -> List[dict]` — each result: `{"score","text","doc_id","chunk_idx","metadata"}`

Examples and a smoke test are provided in `example.py`.

## Tips
- For large corpora, replace the in-memory `VectorStore` with FAISS or hnswlib; current store is brute-force O(N·d).
- If you need exact control of tokenisation/chunk sizes, pass a `tokenizer_id` to `Chunker` matching your LLM.

## Files
- `embedder.py` — embedder implementation and constants.
- `rag.py` — Chunker, VectorStore, and RAGPipeline.
- `example.py` — runnable demo and smoke test.

---
Created for quick integration and prototyping — focused on correct pooling, L2 normalisation, and simple RAG workflows.

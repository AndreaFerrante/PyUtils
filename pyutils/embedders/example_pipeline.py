"""Retrieve JSONL records with LM Studio embeddings and answer with LM Studio chat.

Install the optional JSONL dependency first:
    pip install ".[jsonl]"

Run:
    python -m pyutils.embedders.jsonl_lmstudio_pipeline records.jsonl \
        --embedding-model text-embedding-model \
        --chat-model chat-model \
        --query "What does the data say about VWAP?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None

from pyutils.embedders.rag import Chunk, FAISSStore
from pyutils.lmstudio import LMStudio


CONTEXT_CHAR_LIMIT = 12_000


def _normalise_embeddings(vectors: list[list[float]]) -> np.ndarray:
    """Return contiguous float32 vectors suitable for cosine similarity search."""
    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError("LM Studio embeddings must be a non-empty two-dimensional array.")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("LM Studio returned a zero-norm embedding vector.")
    return np.ascontiguousarray(embeddings / norms[:, None], dtype=np.float32)


def _read_records(
    jsonl_path: Path,
    text_column: str,
    id_column: str | None,
) -> list[tuple[str, str]]:
    """Read valid text records while retaining stable source identifiers."""
    if pl is None:
        raise RuntimeError('Polars is required. Install it with: pip install ".[jsonl]"')
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")

    frame = pl.read_ndjson(jsonl_path)
    if text_column not in frame.columns:
        raise ValueError(f"JSONL text column {text_column!r} was not found.")
    if id_column is not None and id_column not in frame.columns:
        raise ValueError(f"JSONL id column {id_column!r} was not found.")

    records = (
        frame.with_row_index("_row_number")
        .with_columns(pl.col(text_column).cast(pl.String).alias("_text"))
        .filter(pl.col("_text").is_not_null() & (pl.col("_text").str.strip_chars() != ""))
    )

    result: list[tuple[str, str]] = []
    for row in records.iter_rows(named=True):
        source_id = row[id_column] if id_column is not None else row["_row_number"]
        if source_id is None:
            raise ValueError(f"JSONL id column {id_column!r} contains a null value.")
        result.append((str(source_id), row["_text"]))

    if not result:
        raise ValueError("JSONL contains no non-empty values in the text column.")
    return result


def _build_context(sources: list[dict[str, Any]]) -> str:
    """Format retrieved records without exceeding the chat-context budget."""
    parts: list[str] = []
    remaining = CONTEXT_CHAR_LIMIT
    for source in sources:
        header = f"Source ID: {source['doc_id']} (score: {source['score']:.3f})\n"
        available = remaining - len(header)
        if available <= 0:
            break
        text = source["text"][:available]
        parts.append(header + text)
        remaining -= len(header) + len(text) + 2
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def run_pipeline(
    jsonl_path: str | Path,
    *,
    query: str,
    embedding_model: str,
    chat_model: str,
    text_column: str = "text",
    id_column: str | None = None,
    batch_size: int = 64,
    top_k: int = 5,
    client: LMStudio | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Embed JSONL text, retrieve top matches, and answer using retrieved context."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    records = _read_records(Path(jsonl_path), text_column, id_column)
    client = LMStudio() if client is None else client
    store: FAISSStore | None = None

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [text for _, text in batch]
        vectors = client.embed(texts, model=embedding_model)
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            vector_count = len(vectors) if isinstance(vectors, list) else 0
            raise ValueError(
                f"LM Studio returned {vector_count} vectors for {len(texts)} texts."
            )

        embeddings = _normalise_embeddings(vectors)
        if store is None:
            store = FAISSStore(dim=embeddings.shape[1], index_type="flat")

        chunks = [
            Chunk(text=text, doc_id=source_id, chunk_idx=start + index)
            for index, (source_id, text) in enumerate(batch)
        ]
        store.add(chunks, embeddings)

    raw_query_vector = client.embed(query, model=embedding_model)
    if not isinstance(raw_query_vector, list):
        raise ValueError("LM Studio returned an invalid query embedding.")
    query_vector = _normalise_embeddings([raw_query_vector])[0]
    if query_vector.shape[0] != store.dim:
        raise ValueError(
            f"Query embedding dimension {query_vector.shape[0]} does not match "
            f"document dimension {store.dim}."
        )

    sources = store.search(query_vector, top_k=top_k)
    context = _build_context(sources)
    messages = [
        {
            "role": "system",
            "content": "Answer using only the supplied sources. State when they do not answer the question.",
        },
        {
            "role": "user",
            "content": f"Sources:\n{context}\n\nQuestion: {query}",
        },
    ]
    answer = client.chat(messages, model=chat_model)
    return answer, sources


def main() -> None:
    """Run the JSONL retrieval example from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", type=Path, help="Path to a JSONL file.")
    parser.add_argument("--query", required=True, help="Question to answer from retrieved records.")
    parser.add_argument("--embedding-model", required=True, help="Loaded LM Studio embedding model id.")
    parser.add_argument("--chat-model", required=True, help="Loaded LM Studio chat model id.")
    parser.add_argument("--text-column", default="text", help="JSONL text column (default: text).")
    parser.add_argument("--id-column", help="Optional JSONL source identifier column.")
    parser.add_argument("--batch-size", type=int, default=64, help="Texts per embedding request.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved records included in chat context.")
    args = parser.parse_args()

    answer, sources = run_pipeline(
        args.jsonl_path,
        query=args.query,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        text_column=args.text_column,
        id_column=args.id_column,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    print(answer)
    print("\nRetrieved sources:")
    for source in sources:
        print(f"- {source['doc_id']} ({source['score']:.3f})")


if __name__ == "__main__":
    main()
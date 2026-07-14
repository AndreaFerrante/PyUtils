"""Tests for the Polars JSONL to LM Studio retrieval example."""

from pathlib import Path

import polars as pl
import pytest

from PyUtils.pyutils.embedders.example_pipeline import run_pipeline


class FakeLMStudio:
    """Deterministic LM Studio substitute with no server dependency."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.embed_calls: list[tuple[str | list[str], str]] = []
        self.chat_calls: list[tuple[list[dict[str, str]], str]] = []

    def embed(self, text: str | list[str], model: str) -> list[float] | list[list[float]]:
        self.embed_calls.append((text, model))
        if isinstance(text, str):
            return self.vectors[text]
        return [self.vectors[value] for value in text]

    def chat(self, messages: list[dict[str, str]], model: str) -> str:
        self.chat_calls.append((messages, model))
        return "grounded answer"


def write_jsonl(path: Path, data: dict[str, list[object]]) -> Path:
    """Write a small local JSONL fixture."""
    pl.DataFrame(data).write_ndjson(path)
    return path


def test_pipeline_batches_embeds_retrieves_and_chats(tmp_path: Path):
    """Valid rows are embedded, retrieved, and supplied to the chat model."""
    jsonl = write_jsonl(
        tmp_path / "records.jsonl",
        {
            "id": ["alpha-id", "skip-null", "beta-id", "skip-blank"],
            "text": ["alpha", None, "beta", "   "],
        },
    )
    client = FakeLMStudio(
        {
            "alpha": [3.0, 0.0, 0.0],
            "beta": [0.0, 4.0, 0.0],
            "which record": [2.0, 0.0, 0.0],
        }
    )

    answer, sources = run_pipeline(
        jsonl,
        query="which record",
        embedding_model="embedding-model",
        chat_model="chat-model",
        id_column="id",
        batch_size=1,
        top_k=2,
        client=client,
    )

    assert answer == "grounded answer"
    assert [source["doc_id"] for source in sources] == ["alpha-id", "beta-id"]
    assert sources[0]["score"] == pytest.approx(1.0)
    assert client.embed_calls == [
        (["alpha"], "embedding-model"),
        (["beta"], "embedding-model"),
        ("which record", "embedding-model"),
    ]
    messages, model = client.chat_calls[0]
    assert model == "chat-model"
    assert "alpha-id" in messages[1]["content"]
    assert "alpha" in messages[1]["content"]


def test_pipeline_uses_original_row_number_without_id_column(tmp_path: Path):
    """Generated source ids remain stable when blank rows are removed."""
    jsonl = write_jsonl(
        tmp_path / "records.jsonl",
        {"text": ["first", " ", "third"]},
    )
    client = FakeLMStudio(
        {
            "first": [1.0, 0.0],
            "third": [0.0, 1.0],
            "find third": [0.0, 1.0],
        }
    )

    _, sources = run_pipeline(
        jsonl,
        query="find third",
        embedding_model="embedding-model",
        chat_model="chat-model",
        top_k=1,
        client=client,
    )

    assert sources[0]["doc_id"] == "2"


def test_pipeline_rejects_missing_file(tmp_path: Path):
    """Input must point to an existing JSONL file."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_pipeline(
            tmp_path / "missing.jsonl",
            query="query",
            embedding_model="embedding-model",
            chat_model="chat-model",
            client=FakeLMStudio({}),
        )


def test_pipeline_rejects_missing_text_column(tmp_path: Path):
    """Input must provide the configured text column."""
    jsonl = write_jsonl(tmp_path / "records.jsonl", {"body": ["alpha"]})

    with pytest.raises(ValueError, match="text column"):
        run_pipeline(
            jsonl,
            query="query",
            embedding_model="embedding-model",
            chat_model="chat-model",
            client=FakeLMStudio({}),
        )


def test_pipeline_rejects_mismatched_embedding_count(tmp_path: Path):
    """Each source text must have one returned embedding."""
    jsonl = write_jsonl(tmp_path / "records.jsonl", {"text": ["alpha", "beta"]})
    client = FakeLMStudio({"alpha": [1.0, 0.0], "beta": [0.0, 1.0]})
    client.embed = lambda text, model: [[1.0, 0.0]] if isinstance(text, list) else [1.0, 0.0]

    with pytest.raises(ValueError, match="returned 1 vectors for 2 texts"):
        run_pipeline(
            jsonl,
            query="query",
            embedding_model="embedding-model",
            chat_model="chat-model",
            client=client,
        )


def test_pipeline_rejects_zero_norm_embedding(tmp_path: Path):
    """Cosine retrieval cannot use zero-norm embedding vectors."""
    jsonl = write_jsonl(tmp_path / "records.jsonl", {"text": ["alpha"]})
    client = FakeLMStudio({"alpha": [0.0, 0.0], "query": [1.0, 0.0]})

    with pytest.raises(ValueError, match="zero-norm"):
        run_pipeline(
            jsonl,
            query="query",
            embedding_model="embedding-model",
            chat_model="chat-model",
            client=client,
        )
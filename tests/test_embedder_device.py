"""Device-selection tests for the Qwen embedder."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from pyutils.embedders import embedder


def test_auto_device_prefers_cuda():
    """CUDA wins when both CUDA and MPS are available."""
    with patch.object(embedder.torch.cuda, "is_available", return_value=True):
        assert embedder._auto_device() == "cuda"


def test_auto_device_uses_mps_when_cuda_unavailable():
    """Apple Silicon MPS is selected when CUDA is unavailable."""
    mps = SimpleNamespace(is_available=lambda: True)
    with (
        patch.object(embedder.torch.cuda, "is_available", return_value=False),
        patch.object(embedder.torch.backends, "mps", mps),
    ):
        assert embedder._auto_device() == "mps"


def test_auto_device_falls_back_to_cpu():
    """CPU is selected when no accelerator is available."""
    mps = SimpleNamespace(is_available=lambda: False)
    with (
        patch.object(embedder.torch.cuda, "is_available", return_value=False),
        patch.object(embedder.torch.backends, "mps", mps),
    ):
        assert embedder._auto_device() == "cpu"


def test_default_model_is_loaded_and_reports_its_dimension():
    """Default construction preserves the Qwen model and its embedding width."""
    model = MagicMock()
    model.config.hidden_size = embedder.EMBEDDING_DIM
    model.to.return_value = model
    model.eval.return_value = model

    with (
        patch.object(embedder.AutoTokenizer, "from_pretrained") as tokenizer,
        patch.object(embedder.AutoModel, "from_pretrained", return_value=model) as loader,
    ):
        instance = embedder.QwenEmbedder(device="cpu")

    tokenizer.assert_called_once_with(embedder.MODEL_ID, padding_side="left")
    loader.assert_called_once_with(embedder.MODEL_ID)
    assert instance.model_id == embedder.MODEL_ID
    assert instance.dim == embedder.EMBEDDING_DIM


def test_custom_model_is_loaded_and_reports_its_dimension():
    """A supplied model id controls both loaders and the reported vector width."""
    model_id = "Qwen/Qwen3-Embedding-4B"
    model = MagicMock()
    model.config.hidden_size = 2560
    model.to.return_value = model
    model.eval.return_value = model

    with (
        patch.object(embedder.AutoTokenizer, "from_pretrained") as tokenizer,
        patch.object(embedder.AutoModel, "from_pretrained", return_value=model) as loader,
    ):
        instance = embedder.QwenEmbedder(model_id=model_id, device="cpu")

    tokenizer.assert_called_once_with(model_id, padding_side="left")
    loader.assert_called_once_with(model_id)
    assert instance.model_id == model_id
    assert instance.dim == 2560


def test_empty_document_uses_selected_model_dimension():
    """Empty results preserve the vector width of a custom model."""
    model = MagicMock()
    model.config.hidden_size = 2560
    model.to.return_value = model
    model.eval.return_value = model
    encoded = {"input_ids": torch.empty((1, 0), dtype=torch.long)}
    encoded = type("Encoded", (dict,), {"to": lambda self, device: self})(encoded)

    with (
        patch.object(embedder.AutoTokenizer, "from_pretrained") as tokenizer,
        patch.object(embedder.AutoModel, "from_pretrained", return_value=model),
    ):
        tokenizer.return_value.return_value = encoded
        instance = embedder.QwenEmbedder(
            model_id="Qwen/Qwen3-Embedding-4B", device="cpu",
        )
        texts, vectors = instance.encode_document("")

    assert texts == []
    assert vectors.shape == (0, 2560)


def test_example_uses_automatic_device_selection():
    """The runnable example must not force the CUDA-only device."""
    example = Path(__file__).parents[1] / "pyutils" / "embedders" / "example_embedders.py"
    tree = ast.parse(example.read_text())
    embedder_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QwenEmbedder"
    ]

    assert len(embedder_calls) == 1
    assert not embedder_calls[0].keywords
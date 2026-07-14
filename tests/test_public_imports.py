"""Regression tests for documented package-level imports."""


def test_ai_package_exports_are_importable():
    """Public AI clients can be imported through their package namespaces."""
    from pyutils.embedders import FAISSStore, QwenEmbedder, RAGPipeline
    from pyutils.lmstudio import LMStudio, LMStudioError

    assert FAISSStore is not None
    assert QwenEmbedder is not None
    assert RAGPipeline is not None
    assert LMStudio is not None
    assert LMStudioError is not None
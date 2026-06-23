"""
embedder.py — Qwen3-Embedding-0.6B text embedder with late chunking.

Model facts (HuggingFace model card, June 2025):
    Model       : Qwen/Qwen3-Embedding-0.6B (0.6B params, 28 layers)
    Output dim  : 1024  (MRL: truncatable to 32–1024)
    Context     : 32 768 tokens
    Pooling     : last-token pool (EOS position) — decoder architecture
    Norm        : L2 required → cosine ≡ dot product
    Queries     : benefit from "Instruct: {task}\\nQuery: {text}" prefix
    Documents   : no instruction prefix

Two encoding modes:
    encode()           — standard batched encoding, last-token pool
    encode_document()  — late chunking: one forward pass, last-token pool per window

Requirements:
    pip install transformers>=4.51.0 torch numpy
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024
MAX_SEQ_TOKENS = 32_768


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Pool at the last real (non-padding) token per sequence.

    Qwen3-Embedding is a decoder — the EOS token aggregates the full
    sequence.  With left-padding, the last position is always real,
    so we just take hidden_states[:, -1].  The fallback branch handles
    right-padding for completeness.

    Directly from the model card.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1).long() - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, lengths]


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class QwenEmbedder:
    """
    Encodes text → L2-normalised 1024-d float32 vectors.

    Standard mode (queries & short texts):
        vecs = embedder.encode(["query 1", "query 2"], is_query=True)

    Late chunking mode (documents):
        texts, vecs = embedder.encode_document("long document ...", chunk_tokens=512)
        # texts[i] = decoded chunk string
        # vecs.shape = (n_chunks, 1024)

    Args:
        device:     "cuda" | "mps" | "cpu" | None (auto)
        batch_size: texts per forward pass for encode()
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.device = device or _auto_device()
        self.batch_size = batch_size

        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, padding_side="left",
        )
        self._model = AutoModel.from_pretrained(MODEL_ID)
        self._model.to(self.device).eval()

        logger.info("QwenEmbedder ready — device=%s", self.device)

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Standard encode — batched, last-token pool
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: str | List[str],
        is_query: bool = False,
        task: str = "Given a web search query, retrieve relevant passages that answer the query",
    ) -> np.ndarray:
        """
        Encode texts with last-token pooling.

        Args:
            texts:    one string or list of strings
            is_query: if True, prepend the task instruction (improves retrieval 1-5%)
            task:     instruction string (only used when is_query=True)

        Returns:
            (N, 1024) float32, L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]

        if is_query:
            texts = [f"Instruct: {task}\nQuery: {t}" for t in texts]

        parts: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_TOKENS,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out = self._model(**encoded)

            pooled = _last_token_pool(out.last_hidden_state, encoded["attention_mask"])
            normed = F.normalize(pooled, p=2, dim=1)
            parts.append(normed.cpu().numpy())

        return np.vstack(parts).astype(np.float32)

    # ------------------------------------------------------------------
    # Late chunking — one forward pass, contextual chunk embeddings
    # ------------------------------------------------------------------

    def encode_document(
        self,
        text: str,
        chunk_tokens: int = 512,
    ) -> tuple[List[str], np.ndarray]:
        """
        Late chunking: embed a full document, return chunk-level vectors.

        How it works:
            1. Tokenize the full document (up to MAX_SEQ_TOKENS).
            2. One forward pass → one hidden state per token. Under causal
               attention, each token's state already sees every token before
               it — so a token near the end of chunk 2 has "read" chunk 1.
            3. Split the token positions into windows of chunk_tokens.
            4. Represent each window by the hidden state of its LAST token.
            5. L2-normalise.

        Why last-token (not mean) pooling:
            Qwen3-Embedding is trained so the final token's hidden state
            summarises the sequence. Queries are pooled the same way, so
            query and chunk vectors live in the same space and their cosine
            similarity is meaningful. Mean-pooling would (a) mismatch the
            query space and (b) use an aggregation the model never learned.

        Why this still preserves cross-chunk context:
            The last token of each window has causally attended over all
            preceding text, including earlier chunks. So later chunks carry
            context from earlier ones — the whole point of late chunking.

        Args:
            text:          raw document string (truncated at 32k tokens)
            chunk_tokens:  tokens per chunk window (no overlap needed — the
                           model's causal attention provides cross-chunk context)

        Returns:
            (chunk_texts, embeddings) where
                chunk_texts : list[str] — decoded text per chunk
                embeddings  : (n_chunks, 1024) float32, L2-normalised.
                              Empty input → ([], zeros (0, 1024)).
        """
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_TOKENS,
            return_tensors="pt",
        ).to(self.device)

        token_ids = encoded["input_ids"][0]           # (seq_len,)
        seq_len = token_ids.size(0)

        if seq_len == 0:                              # empty / whitespace text
            return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        with torch.no_grad():
            out = self._model(**encoded)
        hidden = out.last_hidden_state[0]             # (seq_len, 1024)

        chunk_texts: List[str] = []
        chunk_vecs: List[torch.Tensor] = []

        for start in range(0, seq_len, chunk_tokens):
            end = min(start + chunk_tokens, seq_len)

            # Represent the window by its last token (matches query pooling).
            chunk_vecs.append(hidden[end - 1])

            # Decode this window's tokens back to readable text.
            ids = token_ids[start:end]
            chunk_texts.append(
                self._tokenizer.decode(ids, skip_special_tokens=True).strip()
            )

        embeddings = torch.stack(chunk_vecs)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return chunk_texts, embeddings.cpu().numpy().astype(np.float32)
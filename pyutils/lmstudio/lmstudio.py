"""
LM Studio REST client.

Covers every documented HTTP endpoint of LM Studio 0.4.x using plain `requests`.
No SDK, no msgspec — so it is immune to the SDK deserialization bug.

Two endpoint families, per the official docs:
  - OpenAI-compatible (`/v1/*`)  -> inference: chat, completion, embeddings
  - Native            (`/api/v1/*`) -> model management: list / load / unload / download

Auth: LM Studio only needs a token if "Require Authentication" is ON.
      Pass it once to the constructor (or set LM_API_TOKEN) and every call uses it.

Model: pass `model=` to the constructor to set the default model for every
       inference call (chat, complete, embed, extract_from_pdf, ...). Any method
       still accepts its own `model=` to override it for that one call.

Docs: https://lmstudio.ai/docs/developer/rest
"""

from __future__ import annotations

import base64
import csv
import json
import os
from typing import Any

import requests


_EXTRACT_SYSTEM_PROMPTS: dict[str, str] = {
    "json": (
        "You extract information from documents. "
        "Reply with only a single valid JSON value and nothing else. "
        "No prose, no explanation, no Markdown code fences."
    ),
    "txt": (
        "You extract information from documents. "
        "Reply with only the extracted information as plain text. "
        "No prose framing, no Markdown code fences."
    ),
    "csv": (
        "You extract information from documents. "
        "Reply with only CSV: a header row followed by data rows. "
        "No prose, no explanation, no Markdown code fences."
    ),
}


def _strip_code_fence(reply: str) -> str:
    """Return `reply` with a wrapping Markdown code fence removed, if present.

    Small models often wrap structured output in ```json ... ``` despite being
    told not to. Handles a bare ``` opener or a language-tagged one. If the text
    is not a complete fenced block it is returned stripped but otherwise intact.

    A reply with prose after the closing fence, or a one-line fence, is NOT
    unwrapped — it falls through to the caller's parser, which then fails loudly.
    """
    text = reply.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


class LMStudioError(RuntimeError):
    """Raised when LM Studio returns an error or is unreachable."""


class LMStudio:
    """A thin, typed wrapper over the LM Studio REST API."""

    def __init__(
        self,
        host: str = "localhost:1234",
        api_token: str | None = None,
        timeout: float = 120.0,
        model: str = "default",
    ) -> None:
        self.host = host
        self.token = api_token or os.getenv("LM_API_TOKEN")
        self.timeout = timeout
        self.model = model                        # default model for every inference call
        self._openai = f"http://{host}/v1"        # inference
        self._native = f"http://{host}/api/v1"    # management
        self._session = requests.Session()

    def _model(self, model: str | None) -> str:
        """Resolve the model for a call: the explicit argument, else `self.model`."""
        return model if model is not None else self.model

    # ------------------------------------------------------------------ #
    # Internal request helper (one place for headers + error handling)   #
    # ------------------------------------------------------------------ #

    def _request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            resp = self._session.request(
                method,
                url,
                json=body,
                headers=headers,
                timeout=self.timeout,
                stream=stream,
            )
        except requests.exceptions.ConnectionError as exc:
            raise LMStudioError(
                f"Cannot reach LM Studio at {self.host}. "
                f"Is the server running (Developer tab -> Start Server)?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LMStudioError(f"Request to {url} timed out after {self.timeout}s.") from exc

        if stream:
            return resp

        # Surface the server's own error message when present.
        if not resp.ok:
            raise LMStudioError(self._error_message(resp))

        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise LMStudioError(self._error_message(resp))
        return data

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        try:
            payload = resp.json()
            err = payload.get("error", payload)
            if isinstance(err, dict):
                return f"HTTP {resp.status_code}: {err.get('message', err)}"
            return f"HTTP {resp.status_code}: {err}"
        except ValueError:
            return f"HTTP {resp.status_code}: {resp.text[:300]}"

    # ------------------------------------------------------------------ #
    # Inference  (OpenAI-compatible /v1/*)                               #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = -1,
    ) -> str:
        """
        Chat completion. `prompt` may be a string (treated as one user turn)
        or a full OpenAI-style messages list. `model` defaults to `self.model`.

        POST /v1/chat/completions
        """
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        body = {
            "model": self._model(model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._request("POST", f"{self._openai}/chat/completions", body)
        return data["choices"][0]["message"]["content"]

    def chat_with_image(
        self,
        prompt: str,
        image: str,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Vision chat. `image` may be a local file path, an http(s) URL, or a
        data: URI. Requires a VLM (e.g. qwen2-vl) loaded in LM Studio.
        `model` defaults to `self.model`.

        POST /v1/chat/completions  (content blocks with image_url)
        """
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": self._image_to_url(image)}},
        ]
        body = {
            "model": self._model(model),
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "stream": False,
        }
        data = self._request("POST", f"{self._openai}/chat/completions", body)
        return data["choices"][0]["message"]["content"]

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
    ) -> str:
        """
        Text completion (non-chat). `model` defaults to `self.model`.

        POST /v1/completions
        """
        body = {
            "model": self._model(model),
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._request("POST", f"{self._openai}/completions", body)
        return data["choices"][0]["text"]

    def embed(
        self,
        text: str | list[str],
        model: str | None = None,
    ) -> list[float] | list[list[float]]:
        """
        Generate embeddings. Returns a single vector for a string input,
        or a list of vectors for a list input. Requires an embedding model.
        `model` defaults to `self.model` (set an embedding model on the client,
        or pass one here).

        POST /v1/embeddings
        """
        body = {"model": self._model(model), "input": text}
        data = self._request("POST", f"{self._openai}/embeddings", body)
        vectors = [item["embedding"] for item in data["data"]]
        return vectors[0] if isinstance(text, str) else vectors

    def chat_stream(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
    ):
        """
        Streaming chat. Yields text fragments as they arrive (SSE).
        `model` defaults to `self.model`.

        POST /v1/chat/completions  (stream=True)
        """
        import json as _json

        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        body = {
            "model": self._model(model),
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        resp = self._request("POST", f"{self._openai}/chat/completions", body, stream=True)
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if not text.startswith("data: "):
                continue
            payload = text[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            chunk = _json.loads(payload)
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def extract_from_pdf(
        self,
        pdf_path: str,
        instruction: str,
        model: str | None = None,
        output_format: str = "json",
        temperature: float = 0.0,
        max_tokens: int = -1,
    ) -> Any:
        """
        Extract information from a PDF with a local LLM. `model` defaults to
        `self.model`.

        The PDF's entire text layer is sent to the chat model in one request,
        prefixed by `instruction`. No retrieval or chunking is performed; if the
        text exceeds the model's context window LM Studio errors and that is
        raised as LMStudioError.

        output_format:
            "json" -> reply parsed with json.loads; returns any JSON value
                      (usually a dict or list, but a bare string/number/bool/
                      null is returned as-is)
            "txt"  -> reply returned as a stripped str
            "csv"  -> reply returned as a CSV str (checked as parseable)

        Raises:
            ValueError        if output_format is not "json" / "txt" / "csv"
            FileNotFoundError if pdf_path does not exist
            LMStudioError     if the PDF has no text layer, the server errors,
                              or a "json"/"csv" reply cannot be parsed
            PdfReadError      (from PyPDF2) if pdf_path is not a readable PDF

        POST /v1/chat/completions  (via self.chat)
        """
        if output_format not in _EXTRACT_SYSTEM_PROMPTS:
            raise ValueError(
                f"output_format must be one of {sorted(_EXTRACT_SYSTEM_PROMPTS)}, "
                f"got {output_format!r}"
            )

        from pyutils.service_factory.pdf import scrape_pdf_content

        text = scrape_pdf_content(pdf_path)
        if not text.strip():
            raise LMStudioError(
                f"PDF has no extractable text: {pdf_path}. "
                f"It may be scanned images that require OCR."
            )

        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPTS[output_format]},
            {"role": "user", "content": f"{instruction}\n\nDocument:\n{text}"},
        ]
        reply = self.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        if output_format == "txt":
            return reply.strip()

        cleaned = _strip_code_fence(reply)

        if output_format == "json":
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise LMStudioError(
                    f"Model reply is not valid JSON: {cleaned[:300]!r}"
                ) from exc

        # output_format == "csv"
        stripped = cleaned.strip()
        try:
            # ponytail: strict=True is the only mode that rejects malformed CSV
            # (unterminated quotes); it still only catches hard parse errors,
            # not structural nonsense like ragged rows.
            rows = list(csv.reader(stripped.splitlines(), strict=True))
        except csv.Error as exc:
            raise LMStudioError(
                f"Model reply is not valid CSV: {stripped[:300]!r}"
            ) from exc
        if not rows:
            raise LMStudioError(
                f"Model returned an empty CSV reply: {reply[:300]!r}"
            )
        return stripped

    # ------------------------------------------------------------------ #
    # Native stateful chat  (/api/v1/chat)                               #
    # ------------------------------------------------------------------ #

    def chat_stateful(
        self,
        text: str,
        model: str | None = None,
        previous_response_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Stateful chat: the server keeps history. Pass back the returned
        response_id as `previous_response_id` to continue the conversation.
        `model` defaults to `self.model`.

        Returns (reply_text, response_id).

        POST /api/v1/chat
        """
        body: dict[str, Any] = {"model": self._model(model), "input": text}
        if previous_response_id:
            body["previous_response_id"] = previous_response_id

        data = self._request("POST", f"{self._native}/chat", body)
        reply = "".join(
            item.get("content", "")
            for item in data.get("output", [])
            if item.get("type") == "message"
        )
        return reply, data.get("response_id", "")

    # ------------------------------------------------------------------ #
    # Model management  (native /api/v1/models/*)                        #
    # ------------------------------------------------------------------ #

    def list_models(self) -> dict[str, Any]:
        """
        List models known to the server (rich state: loaded/not-loaded, arch,
        context length, etc). Returns the parsed JSON; models are under "data".

        GET /api/v1/models
        """
        return self._request("GET", f"{self._native}/models")

    def loaded_instances(self) -> list[str]:
        """Convenience: instance_ids of models currently loaded in memory."""
        data = self.list_models()
        items = data.get("data") or data.get("models") or []
        out: list[str] = []
        for m in items:
            if m.get("state") == "loaded":
                out.append(m.get("instance_id") or m.get("id"))
        return out

    def load_model(
        self,
        model: str,
        context_length: int | None = None,
        ttl: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """
        Load a model into memory. Returns a dict containing `instance_id`
        and `status`. `ttl` is idle seconds before auto-unload.

        POST /api/v1/models/load
        """
        body: dict[str, Any] = {"model": model}
        if context_length is not None:
            body["context_length"] = context_length
        if ttl is not None:
            body["ttl"] = ttl
        body.update(extra)
        return self._request("POST", f"{self._native}/models/load", body)

    def unload_model(self, instance_id: str) -> dict[str, Any]:
        """
        Unload a model instance from memory. NOTE: the key is `instance_id`
        (the value returned by load_model / loaded_instances), not `model`.

        POST /api/v1/models/unload
        """
        return self._request("POST", f"{self._native}/models/unload", {"instance_id": instance_id})

    def unload_all(self) -> list[str]:
        """Unload every currently loaded instance. Returns the ids unloaded."""
        unloaded: list[str] = []
        for instance_id in self.loaded_instances():
            try:
                self.unload_model(instance_id)
                unloaded.append(instance_id)
            except LMStudioError:
                pass  # best-effort; keep going
        return unloaded

    def download_model(self, model: str) -> dict[str, Any]:
        """
        Start downloading a model (catalog id or Hugging Face URL).
        Returns a dict with `job_id` (absent if already downloaded).

        POST /api/v1/models/download
        """
        return self._request("POST", f"{self._native}/models/download", {"model": model})

    def download_status(self, job_id: str) -> dict[str, Any]:
        """
        Check a download job's progress.

        GET /api/v1/models/download/status/{job_id}
        """
        return self._request("GET", f"{self._native}/models/download/status/{job_id}")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _image_to_url(image: str) -> str:
        """Return a value usable in OpenAI image_url: pass URLs/data URIs through,
        base64-encode local files into a data: URI."""
        if image.startswith(("http://", "https://", "data:")):
            return image
        if os.path.isfile(image):
            with open(image, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("utf-8")
            suffix = os.path.splitext(image)[1].lstrip(".").lower() or "jpeg"
            mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
            return f"data:image/{mime};base64,{encoded}"
        raise LMStudioError(f"Image not found and not a URL/data URI: {image}")


if __name__ == "__main__":
    # Quick smoke test against a running LM Studio. Set LM_API_TOKEN if auth is on.
    lm = LMStudio()

    print("Models:")
    for m in lm.list_models().get("data", []):
        print(f"  {m.get('id')}  [{m.get('state')}]")

    print("\nChat:")
    print(lm.chat("In one sentence, what is LM Studio?"))
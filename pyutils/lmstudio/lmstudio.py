"""
LM Studio REST client.

Covers every documented HTTP endpoint of LM Studio 0.4.x using plain `requests`.
No SDK, no msgspec — so it is immune to the SDK deserialization bug.

Two endpoint families, per the official docs:
  - OpenAI-compatible (`/v1/*`)  -> inference: chat, completion, embeddings
  - Native            (`/api/v1/*`) -> model management: list / load / unload / download

Auth: LM Studio only needs a token if "Require Authentication" is ON.
      Pass it once to the constructor (or set LM_API_TOKEN) and every call uses it.

Docs: https://lmstudio.ai/docs/developer/rest
"""

from __future__ import annotations

import base64
import os
from typing import Any

import requests


class LMStudioError(RuntimeError):
    """Raised when LM Studio returns an error or is unreachable."""


class LMStudio:
    """A thin, typed wrapper over the LM Studio REST API."""

    def __init__(
        self,
        host: str = "localhost:1234",
        api_token: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.host = host
        self.token = api_token or os.getenv("LM_API_TOKEN")
        self.timeout = timeout
        self._openai = f"http://{host}/v1"        # inference
        self._native = f"http://{host}/api/v1"    # management
        self._session = requests.Session()

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
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = -1,
    ) -> str:
        """
        Chat completion. `prompt` may be a string (treated as one user turn)
        or a full OpenAI-style messages list.

        POST /v1/chat/completions
        """
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        body = {
            "model": model,
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
        model: str = "default",
        temperature: float = 0.7,
    ) -> str:
        """
        Vision chat. `image` may be a local file path, an http(s) URL, or a
        data: URI. Requires a VLM (e.g. qwen2-vl) loaded in LM Studio.

        POST /v1/chat/completions  (content blocks with image_url)
        """
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": self._image_to_url(image)}},
        ]
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "stream": False,
        }
        data = self._request("POST", f"{self._openai}/chat/completions", body)
        return data["choices"][0]["message"]["content"]

    def complete(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 100,
    ) -> str:
        """
        Text completion (non-chat).

        POST /v1/completions
        """
        body = {
            "model": model,
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
        model: str = "default",
    ) -> list[float] | list[list[float]]:
        """
        Generate embeddings. Returns a single vector for a string input,
        or a list of vectors for a list input. Requires an embedding model.

        POST /v1/embeddings
        """
        body = {"model": model, "input": text}
        data = self._request("POST", f"{self._openai}/embeddings", body)
        vectors = [item["embedding"] for item in data["data"]]
        return vectors[0] if isinstance(text, str) else vectors

    def chat_stream(
        self,
        prompt: str | list[dict[str, Any]],
        model: str = "default",
        temperature: float = 0.7,
    ):
        """
        Streaming chat. Yields text fragments as they arrive (SSE).

        POST /v1/chat/completions  (stream=True)
        """
        import json as _json

        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        body = {
            "model": model,
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

    # ------------------------------------------------------------------ #
    # Native stateful chat  (/api/v1/chat)                               #
    # ------------------------------------------------------------------ #

    def chat_stateful(
        self,
        text: str,
        model: str = "default",
        previous_response_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Stateful chat: the server keeps history. Pass back the returned
        response_id as `previous_response_id` to continue the conversation.

        Returns (reply_text, response_id).

        POST /api/v1/chat
        """
        body: dict[str, Any] = {"model": model, "input": text}
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
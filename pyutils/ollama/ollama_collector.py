import time
import base64
import requests
import argparse
import pandas as pd
from openai import OpenAI
from typing import Any, Dict, Generator, List, Optional, Union


class OllamaCollector:
    """Ollama LLM interface designed for agentic pipelines.

    Uses the OpenAI-compatible /v1/ layer for chat and embeddings;
    hits native /api/ endpoints where the compatibility layer falls short
    (model listing, token counting).
    """

    DEFAULT_HOST    = "http://localhost:11434"
    DEFAULT_MODEL   = "llama3.2"
    DEFAULT_EMBEDDER = "nomic-embed-text"
    DEFAULT_CONTENT = (
        "You are an AI assistant and a professional Python coder with extensive "
        "expertise in machine learning. You work with precision and always take "
        "the time to carefully craft the best possible answer."
    )

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        content: str = "",
        model: str = "",
        embedder: str = "",
        timeout: float = 125.0,
        max_retries: int = 5,
    ):
        self.host        = host.rstrip("/")
        self.timeout     = timeout
        self.max_retries = max_retries
        self.model       = model   if model   else self.DEFAULT_MODEL
        self.embedder    = embedder if embedder else self.DEFAULT_EMBEDDER
        self.content     = content  if content  else self.DEFAULT_CONTENT
        self.client      = self._create_client()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"host={self.host!r}, "
            f"model={self.model!r}, "
            f"embedder={self.embedder!r}, "
            f"timeout={self.timeout!r})"
        )

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _create_client(self) -> OpenAI:
        return OpenAI(
            base_url    = f"{self.host}/v1",
            api_key     = "ollama",          # required by SDK, ignored by Ollama
            max_retries = self.max_retries,
            timeout     = self.timeout,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except OSError as ex:
            raise OSError(f"Could not encode image '{image_path}': {ex}") from ex

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def get_available_models_dataframe(self, timeout: float = 10.0) -> pd.DataFrame:
        """Return locally pulled models as a DataFrame via native /api/tags."""
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=timeout,
            )
            response.raise_for_status()
            models_raw = response.json().get("models", [])

            rows = []
            for m in models_raw:
                rows.append({
                    "model_name":     m.get("name", ""),
                    "model_size_gb":  round(m.get("size", 0) / 1e9, 2),
                    "modified_at":    m.get("modified_at", ""),
                    "parameter_size": m.get("details", {}).get("parameter_size", ""),
                    "quantization":   m.get("details", {}).get("quantization_level", ""),
                    "family":         m.get("details", {}).get("family", ""),
                })

            return (
                pd.DataFrame(rows)
                .sort_values("model_name")
                .reset_index(drop=True)
            )

        except requests.exceptions.RequestException as ex:
            raise RuntimeError(f"Could not reach Ollama at {self.host}: {ex}") from ex

    # ------------------------------------------------------------------
    # Token counting  (native /api/tokenize — model-aware)
    # ------------------------------------------------------------------

    def get_tokens_count(self, text: str, model: str = "") -> int:
        """Count tokens using Ollama's native tokenizer for the target model."""
        target_model = model if model else self.model
        try:
            response = requests.post(
                f"{self.host}/api/tokenize",
                json={"model": target_model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return len(response.json().get("tokens", []))
        except requests.exceptions.RequestException as ex:
            raise RuntimeError(f"Tokenize request failed: {ex}") from ex

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(
        self,
        text_to_embed: Union[str, List[str]],
        embedder: str = "",
        return_full_object: bool = False,
        timer: bool = False,
    ):
        """Return embeddings via /v1/embeddings (OpenAI-compatible)."""
        target_embedder = embedder if embedder else self.embedder

        start = time.time()
        result = self.client.embeddings.create(
            model = target_embedder,
            input = text_to_embed,
        )
        elapsed = time.time() - start

        if timer:
            print(f"Embedding took: {round(elapsed, 4)}s")

        return result if return_full_object else result.data

    # ------------------------------------------------------------------
    # Chat — single query
    # ------------------------------------------------------------------

    def get_answer_given_query(
        self,
        query: str,
        model: str = "",
        timer: bool = False,
    ) -> str:
        """Single-turn query with system prompt. Returns assistant content."""
        target_model = model if model else self.model

        start = time.time()
        response = self.client.chat.completions.create(
            model    = target_model,
            messages = [
                {"role": "system", "content": self.content},
                {"role": "user",   "content": query},
            ],
        )
        elapsed = time.time() - start

        if timer:
            print(f"Answer took: {round(elapsed, 3)}s")

        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Chat — full message history (core agentic primitive)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = "",
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, Generator]:
        """Multi-turn chat accepting an explicit message list.

        Pass the full conversation history for stateful agentic loops.
        With stream=True returns a generator yielding content chunks.
        With tools provided, returns the raw response (may contain tool_calls).
        """
        target_model = model if model else self.model
        kwargs: Dict[str, Any] = {
            "model":    target_model,
            "messages": messages,
            "stream":   stream,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        if stream:
            return self._stream_chunks(response)

        choice = response.choices[0]

        # Tool call path — caller handles tool dispatch
        if tools and choice.finish_reason == "tool_calls":
            return choice.message

        return choice.message.content

    @staticmethod
    def _stream_chunks(response) -> Generator[str, None, None]:
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Structured output  (JSON schema enforcement)
    # ------------------------------------------------------------------

    def get_structured_answer(
        self,
        query: str,
        schema: Dict[str, Any],
        model: str = "",
    ) -> str:
        """Force model to respond as valid JSON matching the given schema."""
        target_model = model if model else self.model

        response = self.client.chat.completions.create(
            model           = target_model,
            messages        = [
                {"role": "system", "content": self.content},
                {"role": "user",   "content": query},
            ],
            response_format = {
                "type":        "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            },
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Vision  (multimodal query)
    # ------------------------------------------------------------------

    def get_answer_with_image(
        self,
        query: str,
        image_path: str,
        model: str = "",
    ) -> str:
        """Send a text+image query to a vision-capable model."""
        target_model  = model if model else self.model
        encoded_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model    = target_model,
            messages = [
                {"role": "system", "content": self.content},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ],
                },
            ],
        )
        return response.choices[0].message.content


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description     = "OllamaCollector — query a local Ollama instance",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = """
Examples:
    python ollama_collector.py -q "Explain transformers in one sentence"
    python ollama_collector.py -q "What is RAG?" -m mistral --host http://remote:11434
        """,
    )
    parser.add_argument("--query",   "-q", required=True,  help="Prompt to submit")
    parser.add_argument("--host",          default=OllamaCollector.DEFAULT_HOST, help="Ollama host URL")
    parser.add_argument("--model",   "-m", default="",     help="Model name (default: llama3.2)")
    parser.add_argument("--content", "-c", default="",     help="System prompt override")
    parser.add_argument("--stream",  "-s", action="store_true", help="Stream output tokens")

    args = parser.parse_args()

    collector = OllamaCollector(
        host    = args.host,
        content = args.content,
        model   = args.model,
    )

    if args.stream:
        for chunk in collector.chat(
            messages=[{"role": "user", "content": args.query}],
            stream=True,
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        print(collector.get_answer_given_query(args.query))


if __name__ == "__main__":
    main()

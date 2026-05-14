"""OllamaCollector — production-ready Ollama client for agentic pipelines.

Uses the native `ollama` Python package for full access to Ollama-native
features unavailable through the OpenAI-compatible layer: callable tools with
auto-docstring descriptions, thinking/reasoning budgets, Pydantic structured
output, model management, token usage on every response, and first-class async.
"""

import argparse
import asyncio
import base64
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Union,
)

import pandas as pd
import ollama
from ollama import AsyncClient, Client


Think = Optional[Union[bool, Literal["low", "medium", "high"]]]
Tools = Optional[List[Union[Callable, Dict[str, Any]]]]


class OllamaCollector:
    """Ollama LLM client for agentic pipelines.

    Sync and async chat, streaming, tool-call loops, structured output,
    vision, embeddings, and model management — all via the native ollama SDK.
    """

    DEFAULT_HOST     = "http://localhost:11434"
    DEFAULT_MODEL    = "llama3.2"
    DEFAULT_EMBEDDER = "nomic-embed-text"
    DEFAULT_CONTENT  = (
        "You are an AI assistant and a professional Python coder with extensive "
        "expertise in machine learning. You work with precision and always take "
        "the time to carefully craft the best possible answer."
    )

    def __init__(
        self,
        host:     str   = DEFAULT_HOST,
        model:    str   = "",
        embedder: str   = "",
        content:  str   = "",
        timeout:  float = 125.0,
    ) -> None:
        self.host     = host
        self.model    = model    or self.DEFAULT_MODEL
        self.embedder = embedder or self.DEFAULT_EMBEDDER
        self.content  = content  or self.DEFAULT_CONTENT
        self.timeout  = timeout

        self._client       = Client(host=self.host, timeout=self.timeout)
        self._async_client = AsyncClient(host=self.host, timeout=self.timeout)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"host={self.host!r}, "
            f"model={self.model!r}, "
            f"embedder={self.embedder!r})"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("utf-8")
        except OSError as ex:
            raise OSError(f"Could not encode image '{image_path}': {ex}") from ex

    def ping(self) -> bool:
        """Return True if the Ollama daemon is reachable."""
        try:
            self._client.list()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def models(self) -> pd.DataFrame:
        """Return locally pulled models as a DataFrame."""
        response = self._client.list()
        rows = []
        for m in response.models or []:
            details = m.details
            rows.append({
                "model_name":    m.model,
                "size_gb":       round((m.size or 0) / 1e9, 2),
                "modified_at":   str(m.modified_at or ""),
                "parameter_size": getattr(details, "parameter_size", "") if details else "",
                "quantization":  getattr(details, "quantization_level", "") if details else "",
                "family":        getattr(details, "family", "") if details else "",
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("model_name").reset_index(drop=True)

    def running(self) -> List[Dict[str, Any]]:
        """Return models currently loaded in VRAM."""
        response = self._client.ps()
        return [
            {
                "model":      m.model,
                "size_vram":  getattr(m, "size_vram", None),
                "expires_at": str(getattr(m, "expires_at", "")),
            }
            for m in (response.models or [])
        ]

    def show(self, model: str = "") -> Dict[str, Any]:
        """Return metadata for a model: template, parameters, capabilities."""
        resp = self._client.show(model or self.model)
        return {
            "modelfile":    resp.modelfile,
            "parameters":   resp.parameters,
            "template":     resp.template,
            "details":      resp.details,
            "capabilities": getattr(resp, "capabilities", None),
        }

    def pull(self, model: str, stream: bool = False) -> None:
        """Pull a model from the Ollama registry. Prints progress when stream=True."""
        if not stream:
            self._client.pull(model)
            return

        for chunk in self._client.pull(model, stream=True):
            status = getattr(chunk, "status", "")
            total  = getattr(chunk, "total",     None)
            done   = getattr(chunk, "completed", None)
            if total and done:
                pct = round(done / total * 100, 1)
                print(f"\r{status}: {pct}%", end="", flush=True)
            else:
                print(f"\r{status}", end="", flush=True)
        print()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        text:     Union[str, List[str]],
        model:    str  = "",
        truncate: bool = True,
    ) -> List[List[float]]:
        """Return embeddings as list[list[float]] for any input size."""
        response = self._client.embed(
            model    = model or self.embedder,
            input    = text,
            truncate = truncate,
        )
        return response.embeddings

    # ------------------------------------------------------------------
    # Sync — single-turn
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        model: str   = "",
        think: Think = None,
        timer: bool  = False,
    ) -> str:
        """Single-turn query with system prompt. Returns assistant text."""
        kwargs: Dict[str, Any] = {
            "model":    model or self.model,
            "messages": [
                {"role": "system", "content": self.content},
                {"role": "user",   "content": query},
            ],
        }
        if think is not None:
            kwargs["think"] = think

        response = self._client.chat(**kwargs)

        if timer:
            secs = round(response.total_duration / 1e9, 3)
            print(f"Answer took: {secs}s  ({response.eval_count} tokens generated)")

        return response.message.content

    # ------------------------------------------------------------------
    # Sync — multi-turn chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model:    str   = "",
        tools:    Tools = None,
        think:    Think = None,
        format:   Any   = None,
    ) -> Union[str, "ollama.Message"]:
        """Multi-turn chat accepting full message history.

        Returns str when the model produces a final answer.
        Returns the raw Message when finish_reason is 'tool_calls' so the
        caller can handle dispatch and continue the loop manually.
        """
        kwargs: Dict[str, Any] = {
            "model":    model or self.model,
            "messages": messages,
        }
        if tools  is not None: kwargs["tools"]  = tools
        if think  is not None: kwargs["think"]  = think
        if format is not None: kwargs["format"] = format

        response = self._client.chat(**kwargs)

        if tools and response.message.tool_calls:
            return response.message

        return response.message.content

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model:    str   = "",
        think:    Think = None,
    ) -> Generator[str, None, None]:
        """Streaming multi-turn chat. Yields content chunks as they arrive.

        Usage::
            for token in collector.stream_chat(messages):
                print(token, end="", flush=True)
        """
        kwargs: Dict[str, Any] = {
            "model":    model or self.model,
            "messages": messages,
            "stream":   True,
        }
        if think is not None:
            kwargs["think"] = think

        for chunk in self._client.chat(**kwargs):
            content = chunk.message.content
            if content:
                yield content

    # ------------------------------------------------------------------
    # Sync — agentic tool loop
    # ------------------------------------------------------------------

    def run_with_tools(
        self,
        query:     str,
        tools:     List[Callable],
        model:     str   = "",
        max_turns: int   = 10,
        think:     Think = None,
    ) -> str:
        """Agentic loop: auto-dispatches tool calls until model returns final text.

        Pass Python callables directly — Ollama uses their docstrings as tool
        descriptions and infers parameter schemas from type annotations.

        Args:
            tools:     Python callables with type-annotated signatures and docstrings.
            max_turns: Safety ceiling on tool-call iterations.

        Raises:
            ValueError:   Model called an unknown tool.
            RuntimeError: Loop exceeded max_turns without a final answer.
        """
        tool_map = {fn.__name__: fn for fn in tools}
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.content},
            {"role": "user",   "content": query},
        ]
        call_kwargs: Dict[str, Any] = {
            "model": model or self.model,
            "tools": tools,
        }
        if think is not None:
            call_kwargs["think"] = think

        for _ in range(max_turns):
            response = self._client.chat(**call_kwargs, messages=messages)
            messages.append(response.message)

            if not response.message.tool_calls:
                return response.message.content

            for call in response.message.tool_calls:
                fn = tool_map.get(call.function.name)
                if fn is None:
                    raise ValueError(f"Model called unknown tool: {call.function.name!r}")
                result = fn(**call.function.arguments)
                messages.append({
                    "role":      "tool",
                    "content":   str(result),
                    "tool_name": call.function.name,
                })

        raise RuntimeError(
            f"run_with_tools exceeded max_turns={max_turns} without a final answer"
        )

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def ask_structured(
        self,
        query:  str,
        schema: Any,
        model:  str = "",
    ) -> str:
        """Force JSON output matching a schema.

        Args:
            schema: Pydantic BaseModel class **or** JSON schema dict.

        Returns:
            Raw JSON string. Parse with e.g. ``MyModel.model_validate_json(result)``.
        """
        fmt = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
        response = self._client.chat(
            model    = model or self.model,
            messages = [
                {"role": "system", "content": self.content},
                {"role": "user",   "content": query},
            ],
            format  = fmt,
            options = {"temperature": 0},
        )
        return response.message.content

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def ask_with_image(
        self,
        query:      str,
        image_path: str,
        model:      str = "",
    ) -> str:
        """Send a text + image query to a vision-capable model."""
        encoded  = self.encode_image(image_path)
        response = self._client.chat(
            model    = model or self.model,
            messages = [
                {"role": "system", "content": self.content},
                {"role": "user", "content": query, "images": [encoded]},
            ],
        )
        return response.message.content

    # ------------------------------------------------------------------
    # Async — single-turn
    # ------------------------------------------------------------------

    async def async_ask(
        self,
        query: str,
        model: str   = "",
        think: Think = None,
    ) -> str:
        """Async single-turn query with system prompt."""
        kwargs: Dict[str, Any] = {
            "model":    model or self.model,
            "messages": [
                {"role": "system", "content": self.content},
                {"role": "user",   "content": query},
            ],
        }
        if think is not None:
            kwargs["think"] = think

        response = await self._async_client.chat(**kwargs)
        return response.message.content

    # ------------------------------------------------------------------
    # Async — multi-turn chat
    # ------------------------------------------------------------------

    async def async_chat(
        self,
        messages: List[Dict[str, Any]],
        model:    str   = "",
        tools:    Tools = None,
        think:    Think = None,
        format:   Any   = None,
    ) -> Union[str, "ollama.Message"]:
        """Async multi-turn chat. Same semantics as sync chat()."""
        kwargs: Dict[str, Any] = {
            "model":    model or self.model,
            "messages": messages,
        }
        if tools  is not None: kwargs["tools"]  = tools
        if think  is not None: kwargs["think"]  = think
        if format is not None: kwargs["format"] = format

        response = await self._async_client.chat(**kwargs)

        if tools and response.message.tool_calls:
            return response.message

        return response.message.content

    async def async_stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model:    str   = "",
        think:    Think = None,
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat. Yields content chunks as they arrive.

        Usage::
            async for token in collector.async_stream_chat(messages):
                print(token, end="", flush=True)
        """
        kwargs: Dict[str, Any] = {
            "model":    model or self.model,
            "messages": messages,
            "stream":   True,
        }
        if think is not None:
            kwargs["think"] = think

        async for chunk in await self._async_client.chat(**kwargs):
            content = chunk.message.content
            if content:
                yield content

    # ------------------------------------------------------------------
    # Async — agentic tool loop
    # ------------------------------------------------------------------

    async def async_run_with_tools(
        self,
        query:     str,
        tools:     List[Callable],
        model:     str   = "",
        max_turns: int   = 10,
        think:     Think = None,
    ) -> str:
        """Async agentic loop. Same semantics as sync run_with_tools()."""
        tool_map = {fn.__name__: fn for fn in tools}
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.content},
            {"role": "user",   "content": query},
        ]
        call_kwargs: Dict[str, Any] = {
            "model": model or self.model,
            "tools": tools,
        }
        if think is not None:
            call_kwargs["think"] = think

        for _ in range(max_turns):
            response = await self._async_client.chat(**call_kwargs, messages=messages)
            messages.append(response.message)

            if not response.message.tool_calls:
                return response.message.content

            for call in response.message.tool_calls:
                fn = tool_map.get(call.function.name)
                if fn is None:
                    raise ValueError(f"Model called unknown tool: {call.function.name!r}")

                result = fn(**call.function.arguments)
                # support async tools transparently
                if asyncio.iscoroutine(result):
                    result = await result

                messages.append({
                    "role":      "tool",
                    "content":   str(result),
                    "tool_name": call.function.name,
                })

        raise RuntimeError(
            f"async_run_with_tools exceeded max_turns={max_turns} without a final answer"
        )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description     = "OllamaCollector — query a local Ollama instance",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = """
Examples:
    python ollama_collector.py -q "Explain transformers in one sentence"
    python ollama_collector.py -q "What is RAG?" -m mistral --host http://remote:11434
    python ollama_collector.py -q "Summarise this" -m llama3.2 --stream
        """,
    )
    parser.add_argument("--query",   "-q", required=True,  help="Prompt to submit")
    parser.add_argument("--host",          default=OllamaCollector.DEFAULT_HOST, help="Ollama host URL")
    parser.add_argument("--model",   "-m", default="",     help="Model name (default: llama3.2)")
    parser.add_argument("--content", "-c", default="",     help="System prompt override")
    parser.add_argument("--stream",  "-s", action="store_true", help="Stream output tokens")

    args      = parser.parse_args()
    collector = OllamaCollector(host=args.host, content=args.content, model=args.model)

    messages = [{"role": "user", "content": args.query}]

    if args.stream:
        for token in collector.stream_chat(messages):
            print(token, end="", flush=True)
        print()
    else:
        print(collector.ask(args.query))


if __name__ == "__main__":
    main()

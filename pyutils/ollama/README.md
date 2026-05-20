# OllamaCollector

A production-grade Python client for [Ollama](https://ollama.com) built for agentic pipelines.

## What

`OllamaCollector` wraps the native Ollama Python SDK into a single, consistent interface that covers the full surface of what Ollama can do: single-turn queries, multi-turn conversations, streaming, autonomous tool-call loops, structured JSON output, vision, and embeddings — all in both sync and async flavours.

It also exposes model management (list, pull, inspect, monitor VRAM) and optional live web search, routing through Ollama's cloud search layer when an API key is present.

## Why

The OpenAI-compatible layer Ollama exposes loses most of what makes Ollama useful locally: native tool schemas derived from Python docstrings, thinking/reasoning budgets, Pydantic-validated structured output, and per-response token usage. This client stays on the native SDK to keep those capabilities available.

The agentic loop exists because calling a model once is rarely enough. Real pipelines need the model to call tools, observe results, and reason again — automatically, with retry on transient failures, with a configurable safety ceiling on iterations, and with hook points for logging or human-in-the-loop approval at each tool invocation. Both the sync and async loops are fully symmetric in capability.

Retry logic with exponential backoff is built in because local Ollama instances time out under load and remote deployments have network variance. Context-window monitoring is built in because silently running out of tokens produces confusing model behaviour that is hard to debug after the fact.

## API Reference

| Method | Description |
|---|---|
| `ask(query, ...)` | Single-turn query; returns assistant text. |
| `async_ask(query, ...)` | Async version of `ask`. |
| `chat(messages, ...)` | Multi-turn chat accepting full message history; returns text or raw `Message` on tool calls. |
| `async_chat(messages, ...)` | Async version of `chat`. |
| `stream_chat(messages, ...)` | Streaming multi-turn chat; yields content chunks as they arrive. |
| `async_stream_chat(messages, ...)` | Async version of `stream_chat`. |
| `run_with_tools(query, tools, ...)` | Agentic loop: auto-dispatches tool calls until model returns a final answer. |
| `async_run_with_tools(query, tools, ...)` | Async version of `run_with_tools`. |
| `ask_structured(query, schema, ...)` | Forces JSON output matching a Pydantic model or JSON schema dict. |
| `ask_with_image(query, image_path, ...)` | Sends a text + image query to a vision-capable model. |
| `embed(text, ...)` | Returns embeddings as `list[list[float]]` for any input size. |
| `models()` | Returns locally pulled models as a DataFrame. |
| `running()` | Returns models currently loaded in VRAM. |
| `show(model)` | Returns metadata for a model: template, parameters, capabilities. |
| `pull(model, ...)` | Pulls a model from the Ollama registry. |
| `ping()` | Returns `True` if the Ollama daemon is reachable. |

## Examples

### Single-turn query with `ask`

```python
from pyutils.ollama.ollama_collector import OllamaCollector

collector = OllamaCollector(
    model="llama3.2",
    content="You are a senior data analyst. Be precise and concise.",
)

answer = collector.ask(
    query="Explain the difference between variance and standard deviation in two sentences.",
    think="low",   # reasoning budget: low / medium / high
    timer=True,    # prints wall-clock time and token count after the call
)

print(answer)
```

`ask` is a stateless single-turn call. It prepends the system prompt automatically, applies retry with exponential backoff on transient network errors, and warns when the context window approaches its configured limit. `think` enables Ollama's native reasoning mode without any extra scaffolding.

---

### Query with live web search

Requires `OLLAMA_API_KEY` set in the environment (free key at https://ollama.com/settings/keys).

```python
import os
from pyutils.ollama.ollama_collector import OllamaCollector

os.environ["OLLAMA_API_KEY"] = "your-key-here"  # or export before running

collector = OllamaCollector(model="llama3.2")

answer = collector.ask(
    query="What are the most significant AI model releases in the last 30 days?",
    web_search=True,
)

print(answer)
```

With `web_search=True`, `ask` internally runs the agentic tool loop: the model decides when to call `web_search` or `web_fetch`, inspects the results, and only then produces its final answer. The caller receives the final text — the intermediate tool calls are transparent. The loop is bounded by `max_turns` (default 10) and retries on network failures the same way a plain `ask` does.

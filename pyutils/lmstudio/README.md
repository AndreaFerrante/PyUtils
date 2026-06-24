# LM Studio Python Client

A single-file, typed Python wrapper over the **LM Studio REST API** (LM Studio 0.4.x).
Built on plain `requests` — no SDK, no `msgspec` — so it is immune to the SDK
deserialization bug (`dictionary update sequence element #0 has length 1`).

It covers **every documented HTTP endpoint**: chat, vision, streaming, text
completion, embeddings, stateful chat, and full model management (list / load /
unload / download).

---

## Why this exists

LM Studio ships an official `lmstudio` Python SDK, but it can raise a `msgspec`
deserialization error on some setups. This client talks to the same server over
HTTP directly, which sidesteps that entirely and keeps the surface area small and
inspectable.

LM Studio exposes two endpoint families, and this client uses the correct base for
each:

| Family | Base URL | Used for |
| --- | --- | --- |
| OpenAI-compatible | `http://<host>/v1/*` | Inference: chat, completion, embeddings |
| Native | `http://<host>/api/v1/*` | Model management + stateful chat |

---

## Requirements

- Python 3.9+
- `requests`
- LM Studio running with the server enabled (Developer tab -> **Start Server**), default `localhost:1234`
- At least one model loaded (or Just-In-Time loading enabled)

```bash
pip install requests
```

Then drop `lmstudio.py` next to your code. That's the whole install.

---

## Authentication

LM Studio only requires a token if **"Require Authentication"** is ON in the
Developer tab. When it is, generate a token there and provide it one of two ways:

```python
LMStudio(api_token="your-token")     # explicit
```
```bash
export LM_API_TOKEN="your-token"      # picked up automatically
```

If auth is OFF, omit the token entirely — the client simply won't send the header.

---

## Quickstart

```python
from lmstudio import LMStudio

lm = LMStudio()  # host="localhost:1234", token from LM_API_TOKEN if present

print(lm.chat("In one sentence, what is LM Studio?", model="qwen2.5-7b-instruct"))
```

Don't know your model id? List them:

```python
for m in lm.list_models()["data"]:
    print(m["id"], m["state"])
```

---

## API reference

Construction:

```python
LMStudio(host="localhost:1234", api_token=None, timeout=120.0)
```

All methods raise `LMStudioError` on a transport failure or a server error (the
exception message carries the server's own error text). Every method takes a
`model` argument that defaults to `"default"`; pass an explicit id from
`list_models()` for predictable behavior.

### Inference (OpenAI-compatible)

#### `chat(prompt, model="default", temperature=0.7, max_tokens=-1) -> str`
`POST /v1/chat/completions`. `prompt` is either a string (one user turn) or a full
OpenAI-style messages list.

```python
lm.chat("Explain TCP in one line.")
lm.chat([
    {"role": "system", "content": "Answer in three words."},
    {"role": "user", "content": "Describe the ocean."},
])
```

#### `chat_with_image(prompt, image, model="default", temperature=0.7) -> str`
`POST /v1/chat/completions` with image content blocks. Requires a vision model
(VLM) loaded. `image` may be a **local file path**, an **http(s) URL**, or a
**`data:` URI**; local files are base64-encoded automatically.

```python
lm.chat_with_image("What's in this?", "/path/photo.jpg", model="qwen2-vl-7b-instruct")
lm.chat_with_image("Describe it", "https://example.com/cat.jpg", model="qwen2-vl-7b-instruct")
```

#### `chat_stream(prompt, model="default", temperature=0.7) -> Iterator[str]`
`POST /v1/chat/completions` with `stream=True`. Yields text fragments as they
arrive.

```python
for chunk in lm.chat_stream("Count to five."):
    print(chunk, end="", flush=True)
```

#### `complete(prompt, model="default", temperature=0.7, max_tokens=100) -> str`
`POST /v1/completions`. Raw (non-chat) text completion.

```python
lm.complete("The capital of Italy is", max_tokens=16)
```

#### `embed(text, model="default") -> list[float] | list[list[float]]`
`POST /v1/embeddings`. Requires an embedding model. A **string** input returns a
single vector; a **list** input returns a list of vectors in the same order.

```python
vec  = lm.embed("hello world")                  # list[float]
vecs = lm.embed(["first", "second"])            # list[list[float]]
```

### Stateful chat (native)

#### `chat_stateful(text, model="default", previous_response_id=None) -> tuple[str, str]`
`POST /api/v1/chat`. The server retains conversation history. Returns
`(reply_text, response_id)`; pass the id back as `previous_response_id` to
continue.

```python
reply, rid = lm.chat_stateful("My name is Ada.")
reply, rid = lm.chat_stateful("What's my name?", previous_response_id=rid)
```

### Model management (native)

#### `list_models() -> dict`
`GET /api/v1/models`. Full inventory with rich state. Models are under the
`"data"` key (each has `id`, `state`, `arch`, context length, etc.).

#### `loaded_instances() -> list[str]`
Convenience filter over `list_models()` — returns the `instance_id`s currently
loaded in memory.

#### `load_model(model, context_length=None, ttl=None, **extra) -> dict`
`POST /api/v1/models/load`. Loads a model and returns a dict containing
`instance_id` and `status`. `ttl` is idle seconds before auto-unload. Extra load
flags (e.g. `flash_attention=True`) pass straight through.

```python
lm.load_model("qwen2.5-7b-instruct", context_length=16384, ttl=300)
```

#### `unload_model(instance_id) -> dict`
`POST /api/v1/models/unload`. **The argument is `instance_id`** (from `load_model`
or `loaded_instances`), *not* the model name.

#### `unload_all() -> list[str]`
Unloads every loaded instance (best-effort). Returns the ids it unloaded.

#### `download_model(model) -> dict`
`POST /api/v1/models/download`. Accepts a catalog id or a Hugging Face URL.
Returns a dict with `job_id` (absent if the model is already downloaded).

#### `download_status(job_id) -> dict`
`GET /api/v1/models/download/status/{job_id}`. Reports progress: `status` is one of
`downloading | paused | completed | failed | already_downloaded`.

```python
job = lm.download_model("ibm/granite-4-micro")
print(lm.download_status(job["job_id"]))
```

---

## Endpoint map

| Method | HTTP | Endpoint |
| --- | --- | --- |
| `chat`, `chat_with_image`, `chat_stream` | POST | `/v1/chat/completions` |
| `complete` | POST | `/v1/completions` |
| `embed` | POST | `/v1/embeddings` |
| `chat_stateful` | POST | `/api/v1/chat` |
| `list_models`, `loaded_instances` | GET | `/api/v1/models` |
| `load_model` | POST | `/api/v1/models/load` |
| `unload_model`, `unload_all` | POST | `/api/v1/models/unload` |
| `download_model` | POST | `/api/v1/models/download` |
| `download_status` | GET | `/api/v1/models/download/status/{job_id}` |

---

## Error handling

```python
from lmstudio import LMStudio, LMStudioError

lm = LMStudio()
try:
    print(lm.chat("hi", model="does-not-exist"))
except LMStudioError as exc:
    print(f"LM Studio said: {exc}")
```

`LMStudioError` is raised for unreachable server, timeouts, non-2xx responses, and
JSON error payloads. The message includes the HTTP status and the server's own
`error.message` when present.

---

## Gotchas / notes

- **`unload_model` takes `instance_id`, not the model name.** If you loaded
  multiple instances of one model, each has a distinct id (e.g.
  `gemma-3-1b` and `gemma-3-1b:2`). Use `loaded_instances()` to see them, or
  `unload_all()` to clear everything.
- **Download status path is `/download/status/{job_id}`** (a common mistake is
  `/download/{job_id}`).
- **Vision needs a VLM**; **embeddings need an embedding model.** Calling these
  against an LLM-only model will error.
- **JIT loading:** if "Just-In-Time model loading" is ON, an inference call for an
  unloaded model will load it on demand (subject to TTL / Auto-Evict). With it OFF,
  load the model first.
- **`model="default"`** is a convenience fallback; for deterministic behavior pass
  an explicit id from `list_models()`.

---

## Verification

The client ships with a mocked test suite that asserts every method's HTTP verb,
URL, request body, and auth header against the official spec — **29/29 checks
pass**, including the two correctness fixes above (unload uses `instance_id`;
download status uses the `/status/{job_id}` path). The mocks patch the HTTP layer,
so the suite runs without a live LM Studio instance.

---

## Files

| File | Purpose |
| --- | --- |
| `lmstudio.py` | The client (`LMStudio` class + `LMStudioError`). |
| `example.py` | Runnable examples for every method. |
| `README.md` | This document. |

---

## References

- LM Studio REST API: https://lmstudio.ai/docs/developer/rest
- OpenAI-compatible endpoints: https://lmstudio.ai/docs/developer/openai-compat
- Model management: https://lmstudio.ai/docs/developer/rest/load

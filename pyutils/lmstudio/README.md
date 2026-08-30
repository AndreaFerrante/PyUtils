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

# set the model once — every inference call uses it
lm = LMStudio(model="qwen2.5-7b-instruct")  # host="localhost:1234", token from LM_API_TOKEN

print(lm.chat("In one sentence, what is LM Studio?"))
print(lm.chat("And in French?", model="some-other-model"))   # override per call
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
LMStudio(host="localhost:1234", api_token=None, timeout=120.0, model="default")
```

`model` is the default model for every inference call (`chat`, `chat_with_image`,
`chat_stream`, `complete`, `embed`, `chat_stateful`, `extract_from_pdf`). Set it
once on the client; any method still accepts its own `model=` to override it for
that one call. It defaults to `"default"` (LM Studio uses whatever is loaded).
Model-management calls (`load_model`, `unload_model`, `download_model`) are
unaffected — their argument is the model to act on, not to run.

All methods raise `LMStudioError` on a transport failure or a server error (the
exception message carries the server's own error text). Pass an explicit id from
`list_models()` for predictable behavior.

### Inference (OpenAI-compatible)

#### `chat(prompt, model=None, temperature=0.7, max_tokens=-1) -> str`
`POST /v1/chat/completions`. `prompt` is either a string (one user turn) or a full
OpenAI-style messages list.

```python
lm.chat("Explain TCP in one line.")
lm.chat([
    {"role": "system", "content": "Answer in three words."},
    {"role": "user", "content": "Describe the ocean."},
])
```

#### `chat_with_image(prompt, image, model=None, temperature=0.7) -> str`
`POST /v1/chat/completions` with image content blocks. Requires a vision model
(VLM) loaded. `image` may be a **local file path**, an **http(s) URL**, or a
**`data:` URI**; local files are base64-encoded automatically.

```python
lm.chat_with_image("What's in this?", "/path/photo.jpg", model="qwen2-vl-7b-instruct")
lm.chat_with_image("Describe it", "https://example.com/cat.jpg", model="qwen2-vl-7b-instruct")
```

#### `chat_stream(prompt, model=None, temperature=0.7) -> Iterator[str]`
`POST /v1/chat/completions` with `stream=True`. Yields text fragments as they
arrive.

```python
for chunk in lm.chat_stream("Count to five."):
    print(chunk, end="", flush=True)
```

#### `complete(prompt, model=None, temperature=0.7, max_tokens=100) -> str`
`POST /v1/completions`. Raw (non-chat) text completion.

```python
lm.complete("The capital of Italy is", max_tokens=16)
```

#### `embed(text, model=None) -> list[float] | list[list[float]]`
`POST /v1/embeddings`. Requires an embedding model. A **string** input returns a
single vector; a **list** input returns a list of vectors in the same order.

```python
vec  = lm.embed("hello world")                  # list[float]
vecs = lm.embed(["first", "second"])            # list[list[float]]
```

#### `extract_from_pdf(pdf_path, instruction, model=None, output_format="json", temperature=0.0, max_tokens=-1) -> Any`

Reads a PDF's text layer (via `scrape_pdf_content`), sends **all of it** plus
`instruction` to the chat model in one `POST /v1/chat/completions` call, and
returns the reply shaped by `output_format`. No retrieval or chunking — if the
document is larger than the model's context window, LM Studio errors and that is
raised as `LMStudioError`.

| `output_format` | Returns | Notes |
| --- | --- | --- |
| `"json"` | any JSON value (usually `dict`/`list`) | `json.loads` of the reply; Markdown code fences are stripped first. Raises `LMStudioError` if the reply is not valid JSON. |
| `"txt"` | `str` | The reply, stripped. |
| `"csv"` | `str` | The reply as CSV text (fences stripped, checked with `csv.reader(strict=True)`). Raises `LMStudioError` on a hard parse error or an empty reply. |

`temperature` defaults to `0.0` for repeatable extraction.

```python
data = lm.extract_from_pdf(
    "invoice.pdf",
    "Extract invoice_number, total, currency, and due_date.",
    model="qwen2.5-7b-instruct",
    output_format="json",
)
# {'invoice_number': 'INV-2025-001', 'total': 1240.5, 'currency': 'EUR', 'due_date': '2025-12-31'}
```

Raises: `ValueError` (bad `output_format`), `FileNotFoundError` (missing PDF),
`LMStudioError` (no text layer, server/model error, unparseable `json`/`csv`
reply). A malformed / non-PDF file raises PyPDF2's own `PdfReadError`.

#### Pipeline: a folder of PDFs → one CSV

There is no dedicated pipeline object — a batch run is a plain loop over
`extract_from_pdf(..., output_format="csv")`, and `pandas` (already a dependency)
merges the per-PDF tables and writes the final CSV.

This example walks every PDF in `statements/`, asks the loaded LLM to pull each
transaction as a CSV row, tags each row with its source file, stacks them into
one table, and saves `transactions.csv`:

```python
import os
from io import StringIO

import pandas as pd

from pyutils.lmstudio import LMStudio, LMStudioError

FOLDER = "statements"
lm = LMStudio()  # host="localhost:1234"; token from LM_API_TOKEN if set

INSTRUCTION = (
    "Extract every transaction in this bank statement. "
    "Return one row per transaction with exactly these columns: "
    "date, description, amount."
)

frames = []
for name in sorted(os.listdir(FOLDER)):
    if not name.lower().endswith(".pdf"):
        continue

    try:
        csv_text = lm.extract_from_pdf(
            os.path.join(FOLDER, name),
            INSTRUCTION,
            model="qwen2.5-7b-instruct",   # any chat model you have loaded
            output_format="csv",
        )
    except LMStudioError as exc:          # no text layer, bad CSV, server error…
        print(f"skipped {name}: {exc}")
        continue

    df = pd.read_csv(StringIO(csv_text))
    df.insert(0, "source_pdf", name)      # column 0: just the PDF file name
    frames.append(df)

if not frames:
    raise SystemExit(f"no transactions extracted from {FOLDER}/*.pdf")

all_transactions = pd.concat(frames, ignore_index=True)
all_transactions.to_csv("transactions.csv", index=False)
print(f"{len(all_transactions)} transactions from {len(frames)} PDFs -> transactions.csv")
```

Notes:

- **One LLM call per PDF.** Each PDF's full text must fit the model's context
  window (no chunking); an over-long PDF raises `LMStudioError` and the loop
  skips it.
- **`pd.concat` aligns by column name.** If one statement's CSV is missing a
  column another has, the rows still stack and the gaps become `NaN` — no crash.
- **The LLM decides the format.** Pin the columns in `INSTRUCTION`, run once, and
  eyeball `transactions.csv` before trusting it. Lower `temperature` is already
  the default (`0.0`).
- **Scanned PDFs won't work** — `extract_from_pdf` needs a real text layer.

### Stateful chat (native)

#### `chat_stateful(text, model=None, previous_response_id=None) -> tuple[str, str]`
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
| `extract_from_pdf` | POST | `/v1/chat/completions` |
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
- **Model resolution:** a method's `model=` wins; otherwise the client's
  `model` (from the constructor, or reassigned later via `lm.model = "..."`)
  is used; that defaults to `"default"`, which lets LM Studio pick whatever is
  loaded. For deterministic behavior set an explicit id from `list_models()`.

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

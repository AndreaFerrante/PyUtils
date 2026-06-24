"""
example.py — runnable examples for every method of the LMStudio client.

Run:
    export LM_API_TOKEN="your-token"     # only if "Require Authentication" is ON
    python example.py

Read-only examples (list, chat, embed, ...) run automatically.
Mutating / heavy examples (load, unload, unload_all, download) are gated behind
RUN_MUTATING so a naive run never unloads your models or pulls gigabytes.

Edit the CONFIG block to match models you actually have loaded.
"""

from __future__ import annotations

from lmstudio import LMStudio, LMStudioError

# --------------------------------------------------------------------------- #
# CONFIG — change these to identifiers from `lm.list_models()`                 #
# --------------------------------------------------------------------------- #
HOST = "localhost:1234"
MODEL = "qwen2.5-7b-instruct"                       # any loaded LLM
VISION_MODEL = "qwen2-vl-7b-instruct"               # a VLM, for chat_with_image
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"  # an embedding model
IMAGE_PATH = "test.jpg"                             # local file, URL, or data URI
DOWNLOAD_MODEL = "ibm/granite-4-micro"              # used only if RUN_MUTATING
RUN_MUTATING = False                                # set True to run load/unload/download


# --------------------------------------------------------------------------- #
# Small runner: prints a header and never lets one failure stop the others.   #
# --------------------------------------------------------------------------- #
def run(label: str, fn) -> None:
    print(f"\n{'-' * 70}\n{label}\n{'-' * 70}")
    try:
        fn()
    except LMStudioError as exc:
        print(f"[skipped] {exc}")
    except Exception as exc:  # noqa: BLE001 — examples should never crash the run
        print(f"[error] {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Inference (OpenAI-compatible /v1/*)                                          #
# --------------------------------------------------------------------------- #
def example_chat(lm: LMStudio) -> None:
    """Simplest case: a single user turn from a string."""
    reply = lm.chat("In one sentence, what is LM Studio?", model=MODEL)
    print(reply)


def example_chat_messages(lm: LMStudio) -> None:
    """Pass a full OpenAI-style messages list (system + user, multi-turn, etc.)."""
    messages = [
        {"role": "system", "content": "You answer in exactly three words."},
        {"role": "user", "content": "Describe the ocean."},
    ]
    print(lm.chat(messages, model=MODEL))


def example_chat_stream(lm: LMStudio) -> None:
    """Stream tokens as they are generated."""
    print("streaming: ", end="", flush=True)
    for fragment in lm.chat_stream("Count from 1 to 5.", model=MODEL):
        print(fragment, end="", flush=True)
    print()


def example_complete(lm: LMStudio) -> None:
    """Raw text completion (non-chat)."""
    print(lm.complete("The capital of Italy is", model=MODEL, max_tokens=16))


def example_embed_single(lm: LMStudio) -> None:
    """Embed one string -> a single vector."""
    vector = lm.embed("hello world", model=EMBED_MODEL)
    print(f"dimensions: {len(vector)}")
    print(f"first 5:    {vector[:5]}")


def example_embed_batch(lm: LMStudio) -> None:
    """Embed a list -> a list of vectors (same order)."""
    vectors = lm.embed(["first text", "second text"], model=EMBED_MODEL)
    print(f"count: {len(vectors)}  each dim: {len(vectors[0])}")


def example_vision(lm: LMStudio) -> None:
    """Send an image with a question. Needs a VLM loaded.
    IMAGE_PATH may be a local file, an http(s) URL, or a data: URI."""
    answer = lm.chat_with_image("What is in this image?", IMAGE_PATH, model=VISION_MODEL)
    print(answer)


# --------------------------------------------------------------------------- #
# Stateful chat (native /api/v1/chat)                                          #
# --------------------------------------------------------------------------- #
def example_stateful(lm: LMStudio) -> None:
    """The server keeps history; continue via the returned response_id."""
    reply1, rid = lm.chat_stateful("My name is Ada. Remember it.", model=MODEL)
    print(f"turn 1: {reply1}")
    reply2, rid = lm.chat_stateful("What is my name?", model=MODEL, previous_response_id=rid)
    print(f"turn 2: {reply2}")


# --------------------------------------------------------------------------- #
# Model management (native /api/v1/models/*)                                   #
# --------------------------------------------------------------------------- #
def example_list_models(lm: LMStudio) -> None:
    """Full model inventory with state, arch, context length, etc."""
    data = lm.list_models()
    for m in data.get("data", []):
        print(f"  {m.get('id')}  [{m.get('state')}]")


def example_loaded_instances(lm: LMStudio) -> None:
    """Just the instance_ids currently in memory."""
    print(lm.loaded_instances() or "(none loaded)")


def example_load(lm: LMStudio) -> None:
    """Load a model with a custom context length and a 5-minute idle TTL."""
    result = lm.load_model(MODEL, context_length=8192, ttl=300)
    print(f"instance_id: {result.get('instance_id')}  status: {result.get('status')}")


def example_unload(lm: LMStudio) -> None:
    """Unload one instance. NOTE: the argument is instance_id, not the model name."""
    print(lm.unload_model(MODEL))


def example_unload_all(lm: LMStudio) -> None:
    """Clear everything from memory."""
    print(f"unloaded: {lm.unload_all() or '(nothing was loaded)'}")


def example_download(lm: LMStudio) -> None:
    """Kick off a download, then poll its status once."""
    job = lm.download_model(DOWNLOAD_MODEL)
    job_id = job.get("job_id")
    print(f"job: {job_id}  status: {job.get('status')}")
    if job_id:
        print(f"poll: {lm.download_status(job_id)}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    lm = LMStudio(host=HOST)  # reads LM_API_TOKEN from the environment if set

    print("=" * 70)
    print("LM STUDIO CLIENT — EXAMPLES")
    print("=" * 70)

    # Read-only / safe
    run("list_models()", lambda: example_list_models(lm))
    run("loaded_instances()", lambda: example_loaded_instances(lm))
    run("chat() — string prompt", lambda: example_chat(lm))
    run("chat() — messages list", lambda: example_chat_messages(lm))
    run("chat_stream()", lambda: example_chat_stream(lm))
    run("complete()", lambda: example_complete(lm))
    run("embed() — single", lambda: example_embed_single(lm))
    run("embed() — batch", lambda: example_embed_batch(lm))
    run("chat_with_image()", lambda: example_vision(lm))
    run("chat_stateful()", lambda: example_stateful(lm))

    # Mutating / heavy — opt in
    if RUN_MUTATING:
        run("load_model()", lambda: example_load(lm))
        run("unload_model()", lambda: example_unload(lm))
        run("unload_all()", lambda: example_unload_all(lm))
        run("download_model() + download_status()", lambda: example_download(lm))
    else:
        print(f"\n{'-' * 70}\nMutating examples skipped (set RUN_MUTATING = True to run "
              f"load / unload / unload_all / download).\n{'-' * 70}")


if __name__ == "__main__":
    main()
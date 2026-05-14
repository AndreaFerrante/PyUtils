# pyutils.web

A collection of self-contained utilities for common web and network tasks.

## Modules

### `password.py` — Secure password generation

Generates random passwords using Python's `secrets` module, which draws from the OS CSPRNG. Exists because `random`-based generators are not cryptographically safe and should never be used for credentials.

### `sucker.py` — Web scraping, local HTTP serving, and email dispatch

Three unrelated but frequently needed one-liners:

- **`simple_scrape`** — Fetches a URL and extracts the first `.content` div. Useful for pulling text out of content-heavy pages without spinning up a full scraping framework.
- **`dummy_web_server`** — Serves the current directory over HTTP. Replaces the need to remember `python -m http.server` arguments and makes the call scriptable.
- **`email_sender`** — Sends a plain-text email over SMTP/STARTTLS. Avoids pulling in a full email library for simple notification use cases.

### `wol.py` — Wake-on-LAN

Sends a WoL magic packet over UDP broadcast to power on a remote machine by MAC address. Exists because machines that need to be woken remotely don't have running services to target — the only channel available is a broadcast at the network layer before the OS is up.

### `youtuber.py` — YouTube transcript extraction

Downloads and formats captions from YouTube videos (text, JSON, SRT) via the `youtube-transcript-api`. Useful for feeding video content into text pipelines — search, summarisation, LLM ingestion — without manual copy-paste or video processing. Supports multiple languages, manual vs. auto-generated transcript preference, and bulk extraction across multiple videos. Also runnable as a CLI tool.

## Why this module exists

These utilities solve problems that come up repeatedly across projects but don't justify pulling in heavy dependencies or writing boilerplate each time. Everything here is intentionally thin: one concern per function, no shared state, no configuration objects.

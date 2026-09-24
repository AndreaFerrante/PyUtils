# Ponytail cleanup (2026-09-24)

- [x] Packaging: pyproject.toml only (drop setup.cfg, setup.py, requirements.txt); drop PyPDF2 + reportlab (pymupdf covers both)
- [x] pdf.py: pymupdf for read/write/render
- [x] youtuber: fix for youtube-transcript-api >= 1.2 (`list_transcripts` removed); drop dead `prefer_manual`
- [x] datetimer: `get_time_month` delegates to `month_reduction`
- [x] compressor7z: reuse `get_txt_files` from compressor; gzip streams instead of reading whole file
- [x] localhost: csv extension check used substring match (extensionless files read as CSV)
- [x] ollama: drop unused `tool_concurrency`; dedupe web tools + tool-call gating
- [x] openai: models list via SDK; reasoned answer delegates
- [x] lmstudio / embedders: small dedupes, stale doc refs

## Review
- Tests: 191 pass (was 191: -2 dead semaphore tests, +2 regression tests for CSV filter and OpenAI model list).
- Youtuber verified live against YouTube (main is broken with installed youtube-transcript-api 1.2.4).
- Wheel builds from pyproject.toml alone; PyPDF2/reportlab gone from deps.
- Net diff: -156 lines.

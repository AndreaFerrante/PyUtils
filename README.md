# PyUtils

A personal Python utility library covering date/time helpers, file system operations, text processing, web tools, data analysis, and AI integrations. Install once, import anywhere.

## Installation

```bash
pip install .
```

```bash
pip uninstall pyutils
```

Requires Python >= 3.11.

---

## Modules

### `pyutils.service_factory`

#### `datetimer` — Date helpers

Common date calculations returned as formatted strings.

```python
from pyutils.service_factory.datetimer import get_today, get_yesterday, get_recent_monday

get_today()                  # '20260514'
get_yesterday(days=3)        # 3 days ago
get_recent_monday()          # most recent Monday (today if Monday)
get_last_monday()            # Monday of the previous week
get_current_month_startdate() # first day of current month
get_current_year_startdate()  # first day of current year
get_time_month(delta_month=2, current_month=5, current_year=2026)  # '202603' → 2 months back
```

---

#### `anonymizer` — NLP-based text anonymization

Masks named entities (names, dates, organizations, locations, etc.) using [spaCy](https://spacy.io/).

```python
from pyutils.service_factory.anonymizer import Anonymizer

anonymizer = Anonymizer()
result = anonymizer.anonymize_text(
    text_language='english',
    text_to_anonymize="John Wayne filed a claim on January 29 2025.",
    spacy_size_model='lg'   # sm / md / lg
)
```

---

#### `compressor` — gzip batch compressor (CLI)

Compresses all `.txt` files in a directory with gzip.

```bash
python -m pyutils.service_factory.compressor /path/to/files --level 6
python -m pyutils.service_factory.compressor ./docs --level 9 --output ./compressed
python -m pyutils.service_factory.compressor ./texts -l 1 --remove-original --dry-run
```

| Flag | Description |
|------|-------------|
| `-l / --level` | Compression level 1–9 (default: 6) |
| `-o / --output` | Output directory (default: same as input) |
| `--remove-original` | Delete source files after compression |
| `--dry-run` | Preview without compressing |

---

#### `compressor7z` — 7zip batch compressor (CLI)

Compresses `.txt` files using 7zip. Supports individual or single-archive mode and multiple formats.

```bash
python -m pyutils.service_factory.compressor7z /path/to/files --level 6
python -m pyutils.service_factory.compressor7z ./docs --format zip --single-archive --archive-name backup
python -m pyutils.service_factory.compressor7z ./texts -l 9 --remove-original
```

| Flag | Description |
|------|-------------|
| `-l / --level` | Compression level 1–9 (default: 6) |
| `-f / --format` | `7z` \| `zip` \| `tar` \| `gzip` (default: `7z`) |
| `--single-archive` | Pack all files into one archive |
| `--archive-name` | Name for the single archive |
| `--remove-original` | Delete source files after compression |
| `--dry-run` | Preview without compressing |

Requires 7zip installed (`brew install p7zip` on macOS).

---

#### `localhost` — File system and process utilities

```python
from pyutils.service_factory.localhost import (
    get_files_in_path,
    get_file_in_root_path,
    get_file_content,
    get_files_timestamps_in_path,
    get_files_zipped_in_folder,
    get_csv_files_in_path_stacked,
    get_xlsx_files_in_path_stacked,
    list_processes,
    close_app
)

get_files_in_path('/data', file_extension='csv')       # list files by extension
get_file_in_root_path('/project', 'config.yaml')       # recursive search by filename
get_file_content('/data', 'query.sql')                 # read file as string
get_files_timestamps_in_path('/logs')                  # files sorted by creation time
get_files_zipped_in_folder('/data', ['csv', 'txt'])    # zip files by extension
get_csv_files_in_path_stacked('/exports', sep=';')     # stack CSVs into a DataFrame
get_xlsx_files_in_path_stacked('/reports')             # stack Excel files into a DataFrame
list_processes()                                        # [{pid, name}, ...]
close_app('chrome')                                    # terminate by name
```

---

#### `pdf` — PDF generation and extraction

```python
from pyutils.service_factory.pdf import pdf_generator_from_text, scrape_pdf_content

pdf_generator_from_text('output.pdf', 'Hello world')  # create PDF from string
text = scrape_pdf_content('document.pdf')             # extract all text from PDF
```

---

### `pyutils.database`

#### `connector` — SQLite backup and restore

```python
from pyutils.database.connector import manage_database

manage_database('app.db', 'app_backup.db')
# Interactive menu: 1) Backup  2) Restore  3) Quit
```

---

### `pyutils.openai`

#### `openai_collector` — OpenAI API wrapper

Wraps chat completions, embeddings, token counting, and model listing.

```python
from pyutils.openai.openai_collector import OpenAICollector

collector = OpenAICollector(api_key='sk-...', model='gpt-4.1')

answer     = collector.get_answer_given_query('Explain VWAP in one sentence.')
embeddings = collector.get_embeddings(['text one', 'text two'])
tokens     = collector.get_tokens_in_string('How many tokens is this?')
models_df  = collector.get_openai_models_dataframe()
```

Also available as a CLI:

```bash
python -m pyutils.openai.openai_collector --query "What is entropy?" --api_key sk-... --model gpt-4.1-nano
```

---

### `pyutils.scheduler`

#### `scheduler` — Thread-based function scheduler

Runs functions at a specific `datetime` using background threads.

```python
from pyutils.scheduler.scheduler import FunctionScheduler
import datetime

scheduler = FunctionScheduler()
scheduler.schedule(my_function, datetime.datetime(2026, 5, 14, 9, 30))
scheduler.start()  # blocks until all scheduled functions complete
```

---

### `pyutils.data`

#### `analysis` — CSV analysis

```python
from pyutils.data.analysis import csv_analyzer

csv_analyzer('/data/sales.csv', separator=',')              # prints full describe()
csv_analyzer('/data/sales.csv', single_column='revenue')   # prints column mean
```

---

### `pyutils.web`

#### `youtuber` — YouTube transcript extractor (CLI)

Downloads and saves transcripts from YouTube videos in text, JSON, or SRT format.

```bash
python -m pyutils.web.youtuber https://www.youtube.com/watch?v=VIDEO_ID
python -m pyutils.web.youtuber VIDEO_ID --language en es --format srt --output transcript.srt
python -m pyutils.web.youtuber VIDEO_ID --info   # list available transcripts only
```

---

#### `sucker` — Web scraping and email utilities

```python
from pyutils.web.sucker import simple_scrape, email_sender, dummy_web_server

simple_scrape('https://example.com')   # scrapes div.content text

email_sender(
    smtp_server='smtp.gmail.com',
    sender_email='you@gmail.com',
    receiver_email='them@gmail.com',
    password='app-password',
    subject='Hello',
    body='Message body'
)

dummy_web_server(port=8080)   # serves current directory over HTTP (dev/debug only)
```

---

#### `wol` — Wake-on-LAN

Sends a magic packet to wake a machine on the local network.

```python
from pyutils.web.wol import send_magic_packet

send_magic_packet('AA:BB:CC:DD:EE:FF')
```

---

#### `password` — Random password generator

```python
from pyutils.web.password import generate_password

generate_password(length=16)   # 'A7b@f9Gh&Zq4#t!U'
```

> Note: uses `random`, not `secrets`. Not suitable for cryptographic use cases.

---

## License

MIT — see [LICENCE](LICENCE).

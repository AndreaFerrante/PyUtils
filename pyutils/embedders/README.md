# Document search with Qwen3 embeddings

A small, readable pipeline that takes your documents, turns them into lists of
numbers that capture meaning, and finds the passages most relevant to a
question. It is built on the `Qwen3-Embedding-0.6B` model and an exact
similarity-search index, and it runs on an ordinary computer — no graphics card
required.

## How the pieces fit together

```
        ┌──────────────┐
 docs → │  embedder.py │ → passages + their vectors ─┐
        └──────────────┘                             │
                                                     ▼
                                             ┌──────────────┐
                                             │    rag.py    │   exact
                                             │ search index │   similarity
                                             └──────────────┘   search
                                                     ▲
        ┌──────────────┐                             │
query → │  embedder.py │ → query vector ─────────────┘ → best passages
        └──────────────┘
```

## Repository layout

```
qwen-rag/
├── README.md           you are here
├── requirements.txt    pinned dependencies (processor-only)
├── embedder.py         turns text into vectors (loads the model)
├── rag.py              cut into passages, store, and search
├── example.py          a runnable demo, processor-only
└── tests/
    └── test_rag.py      checks the math that must be exactly right
```

## Install

```bash
pip install -r requirements.txt
```

The first run downloads the model (about 1.2 GB) from Hugging Face and caches it.

## Quick start

```python
from rag import RAGPipeline

rag = RAGPipeline()                       # loads the model once
rag.index([
    "Your first document, any length...",
    "Your second document...",
])

for hit in rag.query("your question here", top_k=3):
    print(f"[{hit['score']:.3f}] {hit['text'][:80]}")
```

Each result is a small record: a relevance `score`, the matching passage
`text`, and where it came from (`doc_id`, `chunk_idx`).

---

## The ideas, in plain language

### Turning text into vectors

A **vector** here is just a list of numbers that stands for the meaning of a
piece of text. The model reads text and produces these numbers so that texts
about similar things end up with similar numbers. Once text is numbers,
"find related text" becomes "find nearby numbers," which a computer does
quickly.

### Pooling: from many numbers down to one

The model does not read a sentence as one block. It splits the text into small
units called **tokens** (whole words or word fragments) and produces one vector
for every token. But we want a single vector for a whole passage, not one per
token. **Pooling** is the step that folds those many token-vectors into one.

There are two common recipes:

- average all the token-vectors together (called *mean pooling*), or
- take the vector of the final token (called *last-token pooling*).

This model was trained so that the meaning of the whole passage is gathered into
its final token, so we use last-token pooling. Using the wrong recipe gives
worse results — and mixing two recipes produces vectors that cannot be compared
(see the review notes at the end).

### Late chunking: read first, cut later

Long documents have to be cut into smaller passages so a search can point you at
the exact relevant part. The ordinary approach cuts **first** and then turns
each passage into a vector on its own. The drawback is that each passage is read
in isolation. If the third passage says "she signed the agreement" and the name
only appeared in the first passage, the third passage no longer knows who "she"
is — the cut severed the link.

**Late chunking reverses the order.** It feeds the entire document through the
model in one go, so every passage is read while the model is still aware of
everything that came before it. Only afterwards is the result cut into passages.
Each passage's vector therefore carries the surrounding context instead of
standing alone. The word "late" simply means the cutting happens late — after
reading, not before.

### Pooling inside late chunking

After the model has read the whole document, we again have one vector per token,
but now each token's vector reflects its surroundings. For each passage we still
need to fold its token-vectors into one — the same pooling step as before,
applied passage by passage.

Here we take each passage's **final token**, for two reasons. That token has read
everything up to the end of its passage, including earlier passages, so it
carries the cross-passage context that late chunking is meant to preserve. And
it matches how questions are pooled, so a question and a passage are measured the
same way and can be compared fairly.

### The similarity-search index (what FAISS is)

Once every passage is a vector, answering a question means finding the
passage-vectors closest to the question-vector. Checking millions of vectors one
by one is slow. **FAISS is a library built for exactly this job:** it stores
vectors and returns the closest matches quickly.

By default this project uses FAISS's *exact* index, which compares against every
stored vector and never guesses, so results are as accurate as possible. That is
the right choice until the collection grows large. For big collections you can
switch to an *approximate* index that is far faster or far smaller, in exchange
for occasionally missing a close match. Four choices are built in:

- **exact** — checks every vector. Perfect accuracy. The default.
- **clustered** — sorts vectors into groups and only searches the groups nearest
  the question. Much faster on large collections; you choose how many groups to
  check, trading a little accuracy for speed.
- **graph** — links each vector to its neighbours and hops along the links to
  reach close matches in a few steps. Excellent speed and accuracy, at the cost
  of extra memory for the links.
- **compressed** — like the clustered option, but also shrinks each vector into a
  small code instead of storing it in full. Uses far less memory for very large
  collections; the shrinking loses some detail, so it is the least accurate.

You pick one when you create the store; the rest of the pipeline is unchanged.
See `FAISSStore` in `rag.py` for the names and tuning options.

---

## How a question flows through the code

1. `embedder.py` loads the model once.
2. `rag.py` asks the embedder to read each document with late chunking, getting
   back passages and their vectors, and stores them in the search index.
3. For a question, `rag.py` asks the embedder for the question's vector, then
   asks the index for the closest passages.
4. You get the passages back with a score, ready to hand to a language model or
   read directly.

## Sensible defaults and limits

- **Passage size** defaults to 512 tokens (about one page) — large enough to keep
  an idea intact, small enough to point precisely.
- **No overlap** between passages. Late chunking already carries context across
  boundaries, so overlap would only add duplicates that crowd the results.
- **One document, one pass**, capped at about 32,000 tokens (roughly 50 pages).
  Longer documents lose their tail; split them before indexing.
- Reading a long document in a single pass is heavy on a processor-only machine.
  Indexing many long documents will be slow without a graphics card, though it
  still works correctly.

## Running the tests

The tests check the parts that must be exactly right — pooling, the
normalisation step, the windowing, and the search index — without downloading
the model, so they finish in seconds:

```bash
python tests/test_rag.py
```

## Choosing a search index

`FAISSStore` defaults to exact search. To use an approximate index on a large
collection, build the store yourself and hand it to the pipeline:

```python
from rag import RAGPipeline, FAISSStore

store = FAISSStore(dim=1024, index_type="hnsw")   # dim must match the model
rag = RAGPipeline(store=store)
```

`index_type` is one of `"flat"` (exact, the default), `"ivf"` (clustered),
`"hnsw"` (graph), or `"ivfpq"` (compressed). The store buffers vectors and
builds the index once, on the first search, so the clustered and compressed
indexes can train on the whole collection at that point.

## Review notes (fixed)

Defects found and corrected during review:

1. **Mismatched pooling.** Document passages were being *averaged* while
   questions used the *final-token* recipe. Those two recipes place vectors in
   different spaces, so they are not comparable — identical text scored only
   0.70 similarity instead of 1.00. Both sides now use the final-token recipe.
2. **Empty-document crash.** Reading a blank document tried to stack zero
   passages and raised an error. It now returns no passages instead.
3. **Discarded injected index.** Because the store reports its size through the
   standard length operation, an empty store counted as "nothing," and the
   shortcut used to supply a default index silently threw away a custom one.
   The pipeline now keeps whatever index you pass it.

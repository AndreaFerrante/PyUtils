"""
example.py — usage demo (CPU-only, no GPU required)

Install:
    pip install transformers>=4.51.0 torch numpy faiss-cpu

Run:
    python example.py
"""

from embedder import QwenEmbedder
from rag import RAGPipeline

# ------------------------------------------------------------------
# 1. Bare embedder
# ------------------------------------------------------------------
print("=== QwenEmbedder ===\n")
embedder = QwenEmbedder(device="cuda")

# Standard encoding — short texts
vecs = embedder.encode(["hello world", "machine learning"])
print(f"Shape : {vecs.shape}")          # (2, 1024)
print(f"Norms : {(vecs**2).sum(1)}")    # [1.0, 1.0]

# Query mode — instruction prefix added automatically
q = embedder.encode("What is gradient descent?", is_query=True)
print(f"Query : {q.shape}\n")           # (1, 1024)


# ------------------------------------------------------------------
# 2. Late chunking — one document, contextual chunk embeddings
# ------------------------------------------------------------------
print("=== Late chunking ===\n")

doc = (
    "Gamma exposure (GEX) quantifies the dollar gamma that options market-makers "
    "must hedge for every one-percent move in the underlying. When aggregate GEX "
    "is positive, dealers are long gamma: they sell into rallies and buy into dips, "
    "creating a mean-reverting environment. Negative GEX flips this dynamic — "
    "dealers chase momentum, amplifying moves in both directions. The zero-gamma "
    "level is the strike price at which net GEX changes sign. Tracking GEX gives "
    "discretionary traders a structural read on whether the tape is likely to "
    "compress or expand, independent of any directional view."
)

chunk_texts, chunk_vecs = embedder.encode_document(doc, chunk_tokens=64)
print(f"Chunks : {len(chunk_texts)}")
print(f"Vecs   : {chunk_vecs.shape}")
for i, t in enumerate(chunk_texts):
    print(f"  [{i}] {t[:90]}...")
print()


# ------------------------------------------------------------------
# 3. Full RAG pipeline
# ------------------------------------------------------------------
print("=== RAGPipeline ===\n")

docs = [
    # doc 0 — market structure
    (
        "Gamma exposure (GEX) measures the dollar gamma dealers need to hedge per "
        "1% spot move. When aggregate GEX is positive, dealers are long gamma and "
        "sell rallies / buy dips, creating mean-reversion dynamics. Negative GEX "
        "implies dealers are short gamma and must chase momentum, amplifying moves. "
        "The zero-gamma level is the strike where GEX flips sign and is calculated "
        "by sweeping the Black-Scholes gamma curve across open interest at each strike."
    ),
    # doc 1 — positioning data
    (
        "The CFTC Commitments of Traders (COT) Traders in Financial Futures (TFF) "
        "report breaks positions into Dealer/Intermediary, Asset Manager/Institutional, "
        "Leveraged Funds, and Other Reportables. For ES futures (contract code 13874A), "
        "net positioning of Asset Managers is a proxy for institutional directional bias. "
        "A 52-week percentile rank of net longs above 80 indicates crowded-long conditions; "
        "below 20 signals crowded-short positioning."
    ),
    # doc 2 — order flow
    (
        "Cumulative Volume Delta (CVD) accumulates signed trade volume over a session: "
        "positive when aggressive buyers lift the ask, negative when sellers hit the bid. "
        "A divergence between rising price and falling CVD signals absorption — passive "
        "sellers are absorbing buy-side aggression without moving price higher. This is "
        "a key tape-reading signal for intraday ES entries, particularly at VWAP and "
        "prior-session high/low levels where institutional flow tends to cluster."
    ),
    # doc 3 — crowded positions
    ("A crowded long position is when many traders are all betting that "
     "an asset's price will go up at the same time. This creates risk because:"
     "There are lots of people wanting to sell, but not many buyers to absorb their selling"
     "When the trade breaks, everyone tries to exit at once, causing a sharp price drop"
     "A single piece of bad news can flip the whole position from profitable to losing fast"
     "Volatility spikes and you get hit with bad slippage when trying to get out"
     "In markets like options and futures, crowded longs show up as clustering of bullish bets."
     "When they unwind, it usually happens hard and fast because there's no real conviction left—just traders piling in together."
     )
]

rag = RAGPipeline(embedder=embedder, chunk_tokens=512)
rag.index(docs)
print(f"Indexed {len(rag)} chunks\n")

# For a large corpus, swap in an approximate index for speed/memory. Same
# pipeline, same results format — only the retrieval index changes:
#
# from rag import FAISSStore
# store = FAISSStore(dim=embedder.dim, index_type="hnsw")
# rag = RAGPipeline(embedder=embedder, store=store)
# print(f"Indexed {len(rag)} chunks\n")

queries = [
    "How do dealers affect ES price action?",
    "What is a crowded long position?",
    "Explain cumulative delta",
]

for q in queries:
    print(f"Q: {q}")
    for r in rag.query(q, top_k=2):
        print(f"  [{r['score']:.3f}] ({r['doc_id']}) {r['text'][:20]}...")
    print()
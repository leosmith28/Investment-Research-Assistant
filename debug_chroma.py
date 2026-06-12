"""
debug_chroma.py

Standalone ChromaDB diagnostic script. Run from the project root:
    python debug_chroma.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so config can be imported without
# requiring all API keys — we only need the path constants.
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Derive the same absolute path that vector_store.py uses, without importing
# config.settings (which would fail if API keys are not set in .env).
PERSIST_DIR = str(_PROJECT_ROOT / "data" / "chroma_db")
COLLECTION_NAME = "sec_10k_chunks"

client = chromadb.PersistentClient(path=PERSIST_DIR)
ef = DefaultEmbeddingFunction()
collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)

# ── 1. Total document count ───────────────────────────────────────────────
total = collection.count()
print(f"Total documents in '{COLLECTION_NAME}': {total}")

if total == 0:
    print("\nCollection is empty — run the ingest pipeline first:")
    print("  python -m rag.ingest_pipeline --tickers NVDA AAPL MSFT")
    raise SystemExit(0)

# ── 2. First 3 documents with metadata ───────────────────────────────────
print("\n--- First 3 documents ---")
peek = collection.peek(limit=3)
for i, (doc_id, doc, meta) in enumerate(zip(peek["ids"], peek["documents"], peek["metadatas"])):
    print(f"\n[{i}] id       : {doc_id}")
    print(f"    metadata : {meta}")
    preview = doc[:200].replace("\n", " ") if doc else "(empty)"
    print(f"    content  : {preview}{'…' if doc and len(doc) > 200 else ''}")

# ── 3. Test query filtered to NVDA ───────────────────────────────────────
QUERY = "data center revenue"
TICKER = "NVDA"
print(f"\n--- Query: '{QUERY}' | filter: ticker=$eq:{TICKER} ---")
results = collection.query(
    query_texts=[QUERY],
    n_results=min(6, total),
    where={"ticker": {"$eq": TICKER}},
)
ids = results["ids"][0]
metas = results["metadatas"][0]
print(f"Results returned: {len(ids)}")
for i, (doc_id, meta) in enumerate(zip(ids, metas)):
    print(f"  [{i}] {meta}")

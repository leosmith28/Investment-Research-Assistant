"""
test_chroma_write.py

Standalone ChromaDB write/read smoke test. No .env or project imports needed.

    python test_chroma_write.py
"""

import os
import chromadb

PERSIST_DIR = "data/chroma_db"
COLLECTION = "test_collection"

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION)

collection.upsert(
    ids=["test1"],
    documents=["hello world"],
    metadatas=[{"ticker": "AAPL"}],
)

count = collection.count()
print(f"Document count in '{COLLECTION}': {count}")
print(f"Absolute path: {os.path.abspath(PERSIST_DIR)}")

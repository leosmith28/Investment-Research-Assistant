"""
rag/vector_store.py

ChromaDB vector store backed by an explicit PersistentClient.

Write path (add_documents): calls _chroma_client.get_or_create_collection().upsert()
directly, bypassing langchain-chroma's add_texts shim which silently fails when the
collection was created via a pre-built client= argument in langchain-chroma 0.1.x.

Read path (get_vector_store): returns a langchain_chroma.Chroma instance that wraps
the same _chroma_client, used by the retriever for as_retriever() / MMR search.
"""

import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from rag.embedder import get_embeddings

logger = logging.getLogger("vector_store")

path = CHROMA_PERSIST_DIR
print(f"[vector_store] ChromaDB path: {path}")
_chroma_client = chromadb.PersistentClient(path=path)


def get_vector_store() -> Chroma:
    """Return a LangChain Chroma wrapper for retrieval (as_retriever / MMR)."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        client=_chroma_client,
    )


def add_documents(vector_store: Chroma, documents: list[Document]) -> None:
    """
    Write documents to ChromaDB by calling collection.upsert() directly.

    Bypasses langchain_chroma.Chroma.add_texts() to avoid the silent-failure
    path that occurs when the collection is initialised with a pre-built client.
    """
    if not documents:
        print("[vector_store] add_documents called with 0 documents — skipping")
        return

    from uuid import uuid4

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [str(uuid4()) for _ in documents]

    print(f"[vector_store] Embedding {len(texts)} documents...")
    raw = get_embeddings().embed_documents(texts)

    # Normalise to list[list[float]] — handles numpy arrays, 1-D accidental
    # wrapping, and any other shape returned by the embedding function.
    import numpy as np
    def _to_list(e):
        if isinstance(e, np.ndarray):
            e = e.tolist()
        if e and not isinstance(e[0], (list, float, int)):
            e = [float(x) for x in e]
        return e

    if isinstance(raw, np.ndarray):
        embeddings = raw.tolist()
    else:
        embeddings = [_to_list(e) for e in raw]

    # 1-D case: a single embedding mistakenly not wrapped in a list
    if embeddings and isinstance(embeddings[0], float):
        embeddings = [embeddings]

    BATCH_SIZE = 50
    total = len(ids)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    collection = _chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    print(f"[vector_store] Upserting {total} documents into '{CHROMA_COLLECTION_NAME}' "
          f"({n_batches} batches of {BATCH_SIZE})...")

    for i in range(n_batches):
        start, end = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, total)
        try:
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"[vector_store] Batch {i + 1}/{n_batches} complete "
                  f"(docs {start + 1}–{end})")
        except Exception as e:
            import sys
            import traceback
            print(f"[vector_store] Batch {i + 1}/{n_batches} FAILED (docs {start + 1}–{end})")
            print(f"[vector_store] Exception: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
            raise

    count = collection.count()
    print(f"[vector_store] Collection count after upsert: {count}")

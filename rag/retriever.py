"""
rag/retriever.py

LangChain retriever using Maximum Marginal Relevance (MMR).
MMR balances relevance + diversity — avoids returning near-duplicate chunks
from the same passage, which is common in dense 10-K filings.
"""

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import RETRIEVER_FETCH_K, RETRIEVER_K


def get_retriever(
    vector_store: Chroma,
    ticker_filter: Optional[str] = None,
) -> VectorStoreRetriever:
    """
    Return an MMR retriever. If ticker_filter is set, results are scoped to
    that ticker via a ChromaDB metadata filter.
    """
    search_kwargs: dict = {"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K}
    if ticker_filter:
        # ChromaDB 0.5.x requires the explicit operator form; the bare
        # shorthand {"ticker": value} is not reliably forwarded through
        # LangChain's MMR path.
        search_kwargs["filter"] = {"ticker": {"$eq": ticker_filter}}

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

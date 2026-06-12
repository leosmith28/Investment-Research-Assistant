"""
pipeline/query_engine.py

Single entry point for answering a user question end-to-end:
  1. Retrieve relevant 10-K chunks from ChromaDB (scoped to ticker if provided)
  2. Fetch live market data from Alpha Vantage (only if ticker given)
  3. Call Claude with cached system prompt + retrieved context + live data
  4. Return answer and source metadata

The vector store is initialised once at module load time and reused across
all Streamlit renders, avoiding the cost of reloading embeddings per query.
"""

import logging
from typing import Optional

from llm.claude_client import query as claude_query
from market_data.alpha_vantage_client import format_market_summary
from rag.retriever import get_retriever
from rag.vector_store import get_vector_store

logger = logging.getLogger("query_engine")

_vector_store = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        logger.info("Initialising ChromaDB vector store...")
        _vector_store = get_vector_store()
    return _vector_store


def answer(
    question: str,
    ticker: Optional[str] = None,
) -> dict:
    """
    Returns:
        {
            "answer": str,
            "sources": [{"text", "ticker", "section", "filing_date", "content_type"}],
            "market_data": {"summary": str} | {},
            "usage": {"cache_creation_tokens", "cache_read_tokens",
                      "input_tokens", "output_tokens"},
        }
    """
    vs = _get_vector_store()
    retriever = get_retriever(vs, ticker_filter=ticker)

    filter_desc = f'{{"ticker": {{"$eq": "{ticker}"}}}}' if ticker else "none"
    logger.info("Retrieving chunks — filter: %s | question: %s", filter_desc, question[:80])
    docs = retriever.invoke(question)
    logger.info("Retrieved %d chunk(s)", len(docs))
    for i, doc in enumerate(docs):
        logger.debug("  chunk[%d] ticker=%s section=%s len=%d",
                     i, doc.metadata.get("ticker"), doc.metadata.get("section"), len(doc.page_content))

    chunks = [doc.page_content for doc in docs]
    sources = [
        {
            "text": doc.page_content,
            "ticker": doc.metadata.get("ticker", "unknown"),
            "section": doc.metadata.get("section", "unknown"),
            "filing_date": doc.metadata.get("filing_date", "unknown"),
            "content_type": doc.metadata.get("content_type", "text"),
        }
        for doc in docs
    ]

    market_summary = ""
    market_data: dict = {}
    if ticker:
        logger.info("Fetching live market data for %s", ticker)
        market_summary = format_market_summary(ticker)
        market_data = {"summary": market_summary}

    logger.info("Calling Claude (%d chunks, %d chars total context)", len(chunks), sum(len(c) for c in chunks))
    answer_text, usage = claude_query(
        retrieved_chunks=chunks,
        market_data_summary=market_summary,
        question=question,
    )

    logger.info(
        "Usage — cache_creation: %d, cache_read: %d, input: %d, output: %d",
        usage["cache_creation_tokens"],
        usage["cache_read_tokens"],
        usage["input_tokens"],
        usage["output_tokens"],
    )

    return {
        "answer": answer_text,
        "sources": sources,
        "market_data": market_data,
        "usage": usage,
    }

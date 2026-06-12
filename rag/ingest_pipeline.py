"""
rag/ingest_pipeline.py

CLI that orchestrates the full ingestion flow:
    edgar_fetcher → pdf_extractor → chunker → embedder → vector_store

Run once (or re-run to add new tickers — already-ingested tickers are skipped):
    python -m rag.ingest_pipeline --tickers AAPL MSFT NVDA
    python -m rag.ingest_pipeline --tickers AAPL --force   # re-ingest even if present
"""

import argparse
import logging
from pathlib import Path

from config.settings import FILINGS_DIR
from ingestion.chunker import chunk_tables, chunk_text
from ingestion.edgar_fetcher import fetch_10k_for_tickers
from ingestion.pdf_extractor import extract_tables, extract_text
from rag.vector_store import add_documents, get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_pipeline")


def _ticker_already_ingested(vector_store, ticker: str) -> bool:
    """Check whether any documents for this ticker exist in ChromaDB."""
    results = vector_store.get(where={"ticker": {"$eq": ticker}}, limit=1)
    return len(results.get("ids", [])) > 0


def run_pipeline(tickers: list[str], force: bool = False) -> None:
    vector_store = get_vector_store()

    # Step 1: download filings (skips already-downloaded files automatically)
    filing_results = fetch_10k_for_tickers(tickers, output_base=Path(FILINGS_DIR))

    for ticker, result in filing_results.items():
        if result is None:
            logger.warning("Skipping %s — filing download failed", ticker)
            continue

        path: Path = result["path"]
        filing_date: str = result["filing_date"]

        if not force and _ticker_already_ingested(vector_store, ticker):
            logger.info("Skipping %s — already in ChromaDB (use --force to re-ingest)", ticker)
            continue

        logger.info("Extracting text from %s", path.name)
        try:
            text = extract_text(path)
        except Exception as exc:
            logger.error("Text extraction failed for %s: %s", ticker, exc)
            continue

        logger.info("Extracting tables from %s", path.name)
        try:
            tables = extract_tables(path)
        except Exception as exc:
            logger.warning("Table extraction failed for %s (continuing without tables): %s", ticker, exc)
            tables = []

        text_docs = chunk_text(text, ticker=ticker, filing_date=filing_date)
        table_docs = chunk_tables(tables, ticker=ticker, filing_date=filing_date)
        all_docs = text_docs + table_docs

        logger.info(
            "%s: %d text chunks + %d table chunks = %d total",
            ticker,
            len(text_docs),
            len(table_docs),
            len(all_docs),
        )

        add_documents(vector_store, all_docs)
        logger.info("%s ingestion complete", ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest 10-K filings into ChromaDB")
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT NVDA")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if ticker is already in ChromaDB",
    )
    args = parser.parse_args()
    run_pipeline(args.tickers, force=args.force)

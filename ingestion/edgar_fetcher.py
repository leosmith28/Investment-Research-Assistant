"""
ingestion/edgar_fetcher.py

Downloads the most recent 10-K filing for each requested ticker from SEC EDGAR.

Usage (CLI):
    python -m ingestion.edgar_fetcher --tickers AAPL MSFT NVDA
    python -m ingestion.edgar_fetcher --tickers AAPL --output data/filings

EDGAR API notes:
  - Rate limit: <=10 req/s; we enforce a 0.11 s inter-request sleep
  - User-Agent header is REQUIRED by SEC policy; missing it returns 403
  - CIK resolution uses the bulk company_tickers.json (one request covers all tickers)
  - Submissions JSON has parallel arrays in filings.recent; newest entries are first
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from config.settings import EDGAR_USER_AGENT, FILINGS_DIR

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
REQUEST_DELAY = 0.11  # seconds — keeps us under the 10 req/s SEC limit
# EDGAR's recent array caps at 1000 entries; scan all of them — iteration is
# in-memory so there's no extra network cost regardless of how many Form 4s
# a company files between annual reports (MSFT's most recent 10-K is at index 161).
MAX_FILINGS_SCAN = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("edgar_fetcher")


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": EDGAR_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def _get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """Rate-limited GET with exponential back-off on 429."""
    max_retries = 4
    backoff = 1.0

    for attempt in range(max_retries):
        time.sleep(REQUEST_DELAY)
        response = session.get(url, timeout=30, **kwargs)

        if response.status_code == 429:
            wait = backoff * (2**attempt)
            logger.warning(
                "Rate limited (429). Waiting %.1f s before retry %d/%d",
                wait,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response

    raise RuntimeError(f"Exceeded max retries for: {url}")


def load_ticker_cik_map(session: requests.Session) -> dict[str, str]:
    """
    Fetch https://www.sec.gov/files/company_tickers.json once.

    Response format — values are dicts keyed by ordinal string:
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}

    Returns: {"AAPL": "320193", "MSFT": "789019", ...}
    """
    logger.info("Fetching SEC company tickers master list...")
    data = _get(session, COMPANY_TICKERS_URL).json()
    mapping = {entry["ticker"].upper(): str(entry["cik_str"]) for entry in data.values()}
    logger.info("Loaded %d ticker→CIK mappings", len(mapping))
    return mapping


def resolve_cik(ticker: str, ticker_cik_map: dict[str, str]) -> Optional[str]:
    cik = ticker_cik_map.get(ticker.upper())
    if cik is None:
        logger.warning("Ticker %s not found in SEC company tickers list", ticker)
    return cik


def fetch_submissions(session: requests.Session, cik: str) -> dict:
    """
    GET https://data.sec.gov/submissions/CIK{10-digit-padded-cik}.json

    CIK must be zero-padded to exactly 10 digits:
        320193 → "CIK0000320193"
    """
    padded = cik.zfill(10)
    url = f"{SUBMISSIONS_BASE}/CIK{padded}.json"
    logger.info("Fetching submissions for CIK %s", cik)
    return _get(session, url).json()


def find_latest_10k(submissions: dict) -> Optional[dict]:
    """
    Scan filings.recent parallel arrays for the most recent 10-K.

    EDGAR returns arrays sorted newest-first. We accept any form type that
    starts with "10-K" (e.g. "10-K", "10-K405", "10-K/A") and require a
    primary document with a usable extension.

    Returns a dict with accession_number, filing_date, primary_document, form.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    supported = {".htm", ".html", ".pdf", ".txt"}

    for i, form in enumerate(forms[:MAX_FILINGS_SCAN]):
        if not form.startswith("10-K"):
            continue
        doc = primary_docs[i]
        if Path(doc).suffix.lower() in supported:
            return {
                "accession_number": accessions[i],
                "filing_date": dates[i],
                "primary_document": doc,
                "form": form,
            }
        logger.warning(
            "10-K found but primary doc has unusual extension: %s — will try index fallback",
            doc,
        )

    logger.warning("No suitable 10-K found in the %d most recent filings", MAX_FILINGS_SCAN)
    return None


def _accession_to_nodash(accession_number: str) -> str:
    """'0000320193-23-000106' → '000032019323000106'"""
    return accession_number.replace("-", "")


def _fetch_filing_index(
    session: requests.Session, cik: str, accession_nodash: str
) -> list[dict]:
    """
    Fetch the JSON filing index for a submission.

    URL: .../Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}-index.json

    Returns the list from directory.item, each entry has 'name' and 'type' keys.
    """
    url = f"{ARCHIVES_BASE}/{cik}/{accession_nodash}/{accession_nodash}-index.json"
    logger.debug("Fetching filing index: %s", url)
    try:
        data = _get(session, url).json()
        return data.get("directory", {}).get("item", [])
    except Exception as exc:
        logger.error("Failed to fetch filing index: %s", exc)
        return []


def resolve_primary_document(
    session: requests.Session,
    cik: str,
    accession_number: str,
    primary_document_hint: str,
) -> Optional[str]:
    """
    Return the filename of the primary 10-K document to download.

    Strategy:
      1. Trust the hint if it has a supported extension (no extra request)
      2. Fetch the filing index and find the item typed "10-K"
      3. Fall back to first .htm/.html, then first .pdf in the index
    """
    supported = {".htm", ".html", ".pdf"}

    if primary_document_hint and Path(primary_document_hint).suffix.lower() in supported:
        return primary_document_hint

    accession_nodash = _accession_to_nodash(accession_number)
    items = _fetch_filing_index(session, cik, accession_nodash)

    for item in items:
        name = item.get("name", "")
        if item.get("type") == "10-K" and Path(name).suffix.lower() in supported:
            logger.debug("Resolved primary doc via index type match: %s", name)
            return name

    for item in items:
        name = item.get("name", "")
        if Path(name).suffix.lower() in {".htm", ".html"}:
            logger.debug("Resolved primary doc via index HTM fallback: %s", name)
            return name

    for item in items:
        name = item.get("name", "")
        if Path(name).suffix.lower() == ".pdf":
            logger.debug("Resolved primary doc via index PDF fallback: %s", name)
            return name

    logger.error("Cannot resolve primary document for accession %s", accession_number)
    return None


def download_10k(
    session: requests.Session,
    cik: str,
    accession_number: str,
    filename: str,
    output_dir: Path,
) -> Optional[Path]:
    """
    Stream-download the primary 10-K document to output_dir.

    Destination: output_dir/{safe_accession}_{filename}
    Skips download if the file already exists (idempotent re-runs).
    Deletes partial files on failure.
    """
    accession_nodash = _accession_to_nodash(accession_number)
    doc_url = f"{ARCHIVES_BASE}/{cik}/{accession_nodash}/{filename}"
    safe_accession = accession_number.replace("-", "_")
    local_path = output_dir / f"{safe_accession}_{filename}"

    if local_path.exists():
        logger.info("Already downloaded, skipping: %s", local_path)
        return local_path

    logger.info("Downloading 10-K from %s", doc_url)
    try:
        response = session.get(doc_url, stream=True, timeout=120)
        response.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=65536):
                fh.write(chunk)

        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info("Saved %.2f MB → %s", size_mb, local_path)
        return local_path

    except Exception as exc:
        logger.error("Failed to download %s: %s", doc_url, exc)
        if local_path.exists():
            local_path.unlink()
        return None


def fetch_10k_for_tickers(
    tickers: list[str],
    output_base: Path = Path(FILINGS_DIR),
) -> dict[str, Optional[dict]]:
    """
    Main entry point. For each ticker:
      1. Resolve ticker → CIK (bulk company_tickers.json)
      2. Fetch submissions JSON for that CIK
      3. Locate the most recent 10-K in filings.recent
      4. Resolve the primary document filename (with index fallback)
      5. Download to data/filings/{TICKER}/

    Returns {ticker: {"path": Path, "filing_date": str} | None}.
    """
    session = _make_session()
    results: dict[str, Optional[dict]] = {}

    ticker_cik_map = load_ticker_cik_map(session)

    for ticker in tickers:
        ticker = ticker.upper()
        logger.info("── Processing %s ──", ticker)

        try:
            cik = resolve_cik(ticker, ticker_cik_map)
            if cik is None:
                results[ticker] = None
                continue

            submissions = fetch_submissions(session, cik)
            logger.info("Company: %s (CIK %s)", submissions.get("name", ticker), cik)

            filing_meta = find_latest_10k(submissions)
            if filing_meta is None:
                logger.error("No 10-K found for %s", ticker)
                results[ticker] = None
                continue

            logger.info(
                "Found 10-K filed %s (accession: %s)",
                filing_meta["filing_date"],
                filing_meta["accession_number"],
            )

            filename = resolve_primary_document(
                session,
                cik,
                filing_meta["accession_number"],
                filing_meta["primary_document"],
            )
            if filename is None:
                results[ticker] = None
                continue

            local_path = download_10k(
                session,
                cik,
                filing_meta["accession_number"],
                filename,
                output_base / ticker,
            )
            if local_path is None:
                results[ticker] = None
            else:
                results[ticker] = {
                    "path": local_path,
                    "filing_date": filing_meta["filing_date"],
                }

        except Exception as exc:
            logger.error("Unexpected error processing %s: %s", ticker, exc, exc_info=True)
            results[ticker] = None

    succeeded = [t for t, v in results.items() if v is not None]
    failed = [t for t, v in results.items() if v is None]
    logger.info(
        "Done. Succeeded: %s | Failed: %s",
        ", ".join(succeeded) or "none",
        ", ".join(failed) or "none",
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 10-K filings from SEC EDGAR")
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT NVDA")
    parser.add_argument(
        "--output", default=FILINGS_DIR, help=f"Base output directory (default: {FILINGS_DIR})"
    )
    args = parser.parse_args()

    fetch_10k_for_tickers(tickers=args.tickers, output_base=Path(args.output))

"""
ingestion/pdf_extractor.py

Extracts text and tables from 10-K filings.
Supports .htm/.html (via BeautifulSoup + lxml) and .pdf (via pdfplumber).

HTM is preferred — EDGAR 10-Ks post-2009 are inline XBRL (.htm), and
BeautifulSoup strips the XBRL wrapper tags far more cleanly than PDF parsing.
"""

import re
import unicodedata
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup

_SECTION_MAP = {
    "1": "Item 1 - Business",
    "1a": "Item 1A - Risk Factors",
    "1b": "Item 1B - Unresolved Staff Comments",
    "2": "Item 2 - Properties",
    "3": "Item 3 - Legal Proceedings",
    "4": "Item 4 - Mine Safety Disclosures",
    "5": "Item 5 - Market for Common Equity",
    "6": "Item 6 - Selected Financial Data",
    "7": "Item 7 - MD&A",
    "7a": "Item 7A - Quantitative Market Risk",
    "8": "Item 8 - Financial Statements",
    "9": "Item 9 - Changes in Accountants",
    "9a": "Item 9A - Controls and Procedures",
    "9b": "Item 9B - Other Information",
    "10": "Item 10 - Directors and Officers",
    "11": "Item 11 - Executive Compensation",
    "12": "Item 12 - Security Ownership",
    "13": "Item 13 - Certain Relationships",
    "14": "Item 14 - Principal Accountant Fees",
}

_SECTION_RE = re.compile(r"(?i)\bitem\s+(\d+[a-z]?)\b")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _htm_to_text(filepath: Path) -> str:
    raw = filepath.read_bytes()
    soup = BeautifulSoup(raw, "lxml")

    # Remove non-content tags
    for tag in soup(["script", "style", "head", "nav", "footer", "meta", "link"]):
        tag.decompose()

    # Strip XBRL wrapper tags (ix:nonNumeric, xbrli:context, etc.) but keep text
    for tag in soup.find_all(True):
        if ":" in tag.name:
            tag.unwrap()

    return _normalise(soup.get_text(separator="\n"))


def _pdf_to_text(filepath: Path) -> str:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return _normalise("\n\n".join(pages))


def extract_text(filepath: Path) -> str:
    """
    Extract clean body text from a 10-K filing.
    HTM: BeautifulSoup strips XBRL tags, returns normalised visible text.
    PDF: pdfplumber extracts per-page text, joined with double newlines.
    """
    ext = filepath.suffix.lower()
    if ext in {".htm", ".html"}:
        return _htm_to_text(filepath)
    if ext == ".pdf":
        return _pdf_to_text(filepath)
    raise ValueError(f"Unsupported extension: {ext}")


def _table_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    str_rows = [[str(cell or "").strip() for cell in row] for row in rows]
    # Skip empty tables
    if not any(any(cell for cell in row) for row in str_rows):
        return ""
    ncols = max(len(row) for row in str_rows)
    # Pad rows to uniform width
    str_rows = [row + [""] * (ncols - len(row)) for row in str_rows]
    header = "| " + " | ".join(str_rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * ncols) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in str_rows[1:])
    return "\n".join(filter(None, [header, sep, body]))


def _htm_tables(filepath: Path) -> list[str]:
    raw = filepath.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ").strip() for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(cells)
        md = _table_to_markdown(rows)
        if md:
            tables.append(md)
    return tables


def _pdf_tables(filepath: Path) -> list[str]:
    tables = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                md = _table_to_markdown(table)
                if md:
                    tables.append(md)
    return tables


def extract_tables(filepath: Path) -> list[str]:
    """
    Extract tabular data as markdown table strings.
    Returns one string per table; tables with no non-empty cells are skipped.
    """
    ext = filepath.suffix.lower()
    if ext in {".htm", ".html"}:
        return _htm_tables(filepath)
    if ext == ".pdf":
        return _pdf_tables(filepath)
    raise ValueError(f"Unsupported extension: {ext}")


def detect_section(text_block: str) -> str:
    """
    Return the SEC Item label for the first 'Item X' reference in text_block.
    Returns 'unknown' if no match found.
    """
    match = _SECTION_RE.search(text_block)
    if match:
        key = match.group(1).lower()
        return _SECTION_MAP.get(key, f"Item {match.group(1)}")
    return "unknown"

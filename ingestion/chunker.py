"""
ingestion/chunker.py

Splits extracted 10-K text into LangChain Document chunks with metadata.

Strategy:
  1. Split text at SEC Item section boundaries first (preserves section context)
  2. Apply RecursiveCharacterTextSplitter within each section
  3. Inject metadata: ticker, filing_date, section, content_type

Chunk size 1200 / overlap 200 is tuned for 10-K text:
  - Large enough to capture a coherent argument within a section
  - Small enough that the embedding stays specific to one topic
  - 200-token overlap preserves context across boundaries
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from ingestion.pdf_extractor import _SECTION_MAP, _SECTION_RE

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Matches "Item 7." or "ITEM 1A" at start of a line (section boundary)
_BOUNDARY_RE = re.compile(r"(?:^|\n)(?:ITEM|Item)\s+\d+[A-Za-z]?\b", re.MULTILINE)


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text at Item X headings.
    Returns [(section_label, section_text), ...].
    Pre-heading text is labelled 'Preamble'.
    """
    boundaries = [(m.start(), m.group().strip()) for m in _BOUNDARY_RE.finditer(text)]
    if not boundaries:
        return [("Preamble", text)]

    sections: list[tuple[str, str]] = []

    # Text before the first section heading
    pre = text[: boundaries[0][0]].strip()
    if pre:
        sections.append(("Preamble", pre))

    for i, (start, header) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end]
        m = _SECTION_RE.search(header)
        key = m.group(1).lower() if m else "unknown"
        label = _SECTION_MAP.get(key, f"Item {m.group(1)}" if m else "unknown")
        sections.append((label, section_text))

    return sections


def chunk_text(
    text: str,
    ticker: str,
    filing_date: str,
    section: str = "unknown",
) -> list[Document]:
    """
    Split raw text into overlapping chunks, one LangChain Document per chunk.
    Sections are auto-detected; the `section` parameter is used only as a
    fallback label when detection yields nothing.
    """
    sections = _split_by_sections(text)
    docs: list[Document] = []

    for section_label, section_text in sections:
        for chunk in _SPLITTER.split_text(section_text):
            if not chunk.strip():
                continue
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "section": section_label,
                        "content_type": "text",
                    },
                )
            )

    return docs


def chunk_tables(
    tables: list[str],
    ticker: str,
    filing_date: str,
    section: str = "unknown",
) -> list[Document]:
    """
    Wrap each markdown table as a single Document chunk.
    Tables are kept whole; very large tables should be summarised upstream
    before calling this function.
    """
    return [
        Document(
            page_content=table,
            metadata={
                "ticker": ticker,
                "filing_date": filing_date,
                "section": section,
                "content_type": "table",
            },
        )
        for table in tables
        if table.strip()
    ]

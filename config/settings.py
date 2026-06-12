import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_TOKENS: int = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))

# ── Alpha Vantage ─────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY: str = os.environ["ALPHA_VANTAGE_API_KEY"]
MARKET_DATA_CACHE_TTL: int = int(os.getenv("MARKET_DATA_CACHE_TTL", "60"))

# ── EDGAR ─────────────────────────────────────────────────────────────────
EDGAR_USER_AGENT: str = os.environ["EDGAR_USER_AGENT"]

# ── Paths ─────────────────────────────────────────────────────────────────
FILINGS_DIR: str = os.getenv("FILINGS_DIR", "data/filings")

# os.path.abspath resolves relative to the process CWD at import time.
# This matches exactly what test_chroma_write.py and debug_chroma.py do,
# and avoids any Path(__file__) / symlink / case-normalisation mismatch
# that can occur on Windows/MSYS2.
CHROMA_PERSIST_DIR: str = os.path.abspath(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))

CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "sec_10k_chunks")

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "6"))
RETRIEVER_FETCH_K: int = int(os.getenv("RETRIEVER_FETCH_K", "20"))

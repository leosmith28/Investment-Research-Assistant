"""
app/main.py

Streamlit chat interface for the investment research assistant.

Run:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.components.market_sidebar import render_market_sidebar
from app.components.source_viewer import render_sources
from pipeline.query_engine import answer

st.set_page_config(
    page_title="Investment Research Assistant",
    page_icon="📈",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ticker" not in st.session_state:
    st.session_state.ticker = ""

# ── Sidebar controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    ticker_input = (
        st.text_input(
            "Filter by ticker",
            value=st.session_state.ticker,
            placeholder="AAPL, MSFT, NVDA …",
            help="Leave blank to search across all ingested filings",
        )
        .upper()
        .strip()
    )
    st.session_state.ticker = ticker_input

    st.caption("Ingested tickers: AAPL · MSFT · NVDA")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Live market data renders inside sidebar when a ticker is active
if st.session_state.ticker:
    render_market_sidebar(st.session_state.ticker)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("Investment Research Assistant")
st.caption(
    "Ask questions about SEC 10-K filings · Live market data via Alpha Vantage · Powered by Claude"
)

# ── Replay prior turns ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            if sources:
                st.caption(f"📑 {len(sources)} source passage(s) from 10-K")
                render_sources(sources)
            usage = msg.get("usage")
            if usage:
                with st.expander("🔢 Token usage"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Cache write", usage.get("cache_creation_tokens", 0))
                    c2.metric("Cache read", usage.get("cache_read_tokens", 0))
                    c3.metric("Input", usage.get("input_tokens", 0))
                    c4.metric("Output", usage.get("output_tokens", 0))

# ── Chat input ─────────────────────────────────────────────────────────────
ticker_ctx = f" [{st.session_state.ticker}]" if st.session_state.ticker else ""
prompt = st.chat_input(f"Ask a question about{ticker_ctx} the 10-K filings…")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving filings and querying Claude…"):
            result = answer(
                question=prompt,
                ticker=st.session_state.ticker or None,
            )

        st.markdown(result["answer"])

        sources = result["sources"]
        st.caption(f"📑 {len(sources)} source passage(s) from 10-K")
        render_sources(sources)

        usage = result["usage"]
        with st.expander("🔢 Token usage"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cache write", usage.get("cache_creation_tokens", 0))
            c2.metric("Cache read", usage.get("cache_read_tokens", 0))
            c3.metric("Input", usage.get("input_tokens", 0))
            c4.metric("Output", usage.get("output_tokens", 0))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": sources,
            "usage": usage,
        }
    )

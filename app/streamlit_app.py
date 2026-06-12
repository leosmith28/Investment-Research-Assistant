"""
app/streamlit_app.py

Streamlit frontend for the investment research assistant.

Run:
    streamlit run app/streamlit_app.py
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

st.title("Investment Research Assistant")
st.caption("RAG over SEC 10-K filings + live market data via Alpha Vantage")

ticker = st.text_input("Ticker (optional — leave blank to search all)", value="").upper() or None
question = st.text_area("Research question", placeholder="What are the main risk factors for AAPL?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving filings and querying Claude…"):
        result = answer(question=question, ticker=ticker)

    st.markdown(result["answer"])

    with st.expander("Source passages from 10-K"):
        render_sources(result["sources"])

    if ticker:
        render_market_sidebar(ticker)

"""
app/components/source_viewer.py

Streamlit expander showing retrieved 10-K chunks with metadata labels.
"""

import streamlit as st


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No source passages retrieved.")
        return

    for i, source in enumerate(sources, 1):
        ticker = source.get("ticker", "?")
        section = source.get("section", "unknown")
        filing_date = source.get("filing_date", "unknown")
        content_type = source.get("content_type", "text")
        icon = "📊" if content_type == "table" else "📄"

        with st.expander(f"{icon} Source {i} — {ticker} | {section} | {filing_date}"):
            text = source.get("text", "")
            if content_type == "table":
                st.markdown(text)
            else:
                st.text(text)

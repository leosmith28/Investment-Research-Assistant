"""
app/components/market_sidebar.py

Streamlit sidebar widget: price, change %, P/E, EPS, 52-week range, market cap,
and recent news headlines with sentiment labels.
"""

import streamlit as st

from market_data.alpha_vantage_client import get_fundamentals, get_news_sentiment, get_quote


def render_market_sidebar(ticker: str) -> None:
    with st.sidebar:
        st.subheader(f"📈 {ticker} — Live Market Data")

        quote = get_quote(ticker)
        if quote:
            delta_str = f"{quote['change']} ({quote['change_percent']})"
            st.metric(label="Price", value=f"${quote['price']}", delta=delta_str)
        else:
            st.warning("Quote unavailable")

        fundamentals = get_fundamentals(ticker)
        if fundamentals:
            st.markdown("**Fundamentals**")
            mc = fundamentals["market_cap"]
            try:
                mc_display = f"${float(mc) / 1e9:.2f}B"
            except (ValueError, TypeError):
                mc_display = str(mc)

            col1, col2 = st.columns(2)
            col1.metric("P/E", fundamentals["pe_ratio"])
            col2.metric("EPS", f"${fundamentals['eps']}")
            col1.metric("Fwd P/E", fundamentals["forward_pe"])
            col2.metric("Div Yield", fundamentals["dividend_yield"])
            st.caption(f"Market Cap: {mc_display}")
            st.caption(
                f"52W Range: ${fundamentals['week_52_low']} – ${fundamentals['week_52_high']}"
            )

        news = get_news_sentiment(ticker)
        if news:
            st.markdown("**Recent News**")
            for item in news[:5]:
                label = item["sentiment_label"]
                if "Bullish" in label:
                    emoji = "🟢"
                elif "Bearish" in label:
                    emoji = "🔴"
                else:
                    emoji = "⚪"
                date = item["published"][:8] if len(item["published"]) >= 8 else item["published"]
                title = item["title"]
                if len(title) > 75:
                    title = title[:72] + "…"
                st.caption(f"{emoji} [{date}] {title}")

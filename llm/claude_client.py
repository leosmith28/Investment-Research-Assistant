"""
llm/claude_client.py

Anthropic SDK wrapper with prompt caching.

Cache layout per call:
  - System prompt: cache_control ttl=1h (static across all queries in a session)
  - Retrieved 10-K chunks: cache_control ephemeral/5-min TTL
    (chunks for the same ticker repeat on follow-up questions)
  - Live market data + question: no cache (dynamic tail, ~300 tokens)

Cache hit on the system prompt + chunks reduces cost by ~90% on follow-up
questions about the same ticker.
"""

import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL
from llm.prompt_templates import SYSTEM_PROMPT, build_user_prompt

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def query(
    retrieved_chunks: list[str],
    market_data_summary: str,
    question: str,
) -> tuple[str, dict]:
    """
    Send a query to Claude with retrieved context and live market data.

    Returns (answer_text, usage_dict).
    usage_dict keys: cache_creation_tokens, cache_read_tokens,
                     input_tokens, output_tokens.
    """
    chunks_text = "\n\n---\n\n".join(retrieved_chunks)

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"## Retrieved 10-K Context\n\n{chunks_text}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"## Live Market Data\n\n{market_data_summary}\n\n"
                            f"## Question\n\n{question}"
                        ),
                    },
                ],
            }
        ],
    )

    usage = {
        "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response.content[0].text, usage

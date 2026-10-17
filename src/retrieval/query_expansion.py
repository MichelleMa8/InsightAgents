"""
LLM query expansion for offline seed retrieval (off unless
QUERY_EXPANSION_ENABLED): splits a compound topic into up to NUM_SEARCH_QUERIES
short, single-meaning search queries. The returned list always starts with the
topic itself, and unusable LLM output falls back to the topic alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

import config as cfg
from json_llm import parse_json_response
from llm_client import LLMClient
from prompts.retrieval_prompts import (
    QUERY_EXPANSION_SYSTEM,
    QUERY_EXPANSION_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 120   # anything longer is a sentence, not a query — drop it


def _normalise(query: str) -> str:
    return " ".join(query.lower().split())


def _clean_queries(raw: Any, topic: str, limit: int) -> List[str]:
    """
    Keep at most `limit` usable queries: strings only, whitespace collapsed,
    surrounding quotes stripped, no duplicates (case/whitespace-insensitive)
    and nothing equal to the topic — the topic is always searched anyway, so a
    restatement of it would just spend a slot on a search already being run.
    """
    if not isinstance(raw, list):
        return []
    cleaned: List[str] = []
    seen = {_normalise(topic)}
    for item in raw:
        if not isinstance(item, str):
            continue
        query = " ".join(item.split()).strip("\"' ")
        if not query or len(query) > MAX_QUERY_CHARS:
            continue
        key = _normalise(query)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
        if len(cleaned) >= limit:
            break
    return cleaned


def expand_topic_queries(
    topic: str,
    llm_client: LLMClient,
    num_queries: Optional[int] = None,
) -> List[str]:
    """
    Return the search queries for `topic`: the topic itself, followed by up to
    `num_queries` (default `cfg.NUM_SEARCH_QUERIES`) LLM-generated
    single-meaning queries. Returns `[topic]` when expansion is disabled or the
    LLM never produced a usable query list.
    """
    n = cfg.NUM_SEARCH_QUERIES if num_queries is None else num_queries
    if not cfg.QUERY_EXPANSION_ENABLED or n <= 0:
        return [topic]

    user_msg = QUERY_EXPANSION_USER_TEMPLATE.format(topic=topic, num_queries=n)
    total_attempts = 1 + cfg.QUERY_EXPANSION_MAX_RETRIES
    for attempt in range(1, total_attempts + 1):
        response = llm_client.complete(
            system_prompt=QUERY_EXPANSION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=None,
            max_tokens=cfg.QUERY_EXPANSION_MAX_TOKENS,
            stage="query_expansion",
        )
        raw_text = response.text or ""
        try:
            parsed = parse_json_response(raw_text)
        except json.JSONDecodeError as e:
            logger.warning(
                "Query expansion attempt %d/%d: unparseable response (%s). Raw: %s",
                attempt, total_attempts, e, raw_text[:300],
            )
            continue
        queries = _clean_queries(parsed.get("queries"), topic, n)
        if queries:
            logger.info("Expanded topic '%s' into %d search quer%s: %s",
                        topic, len(queries), "y" if len(queries) == 1 else "ies", queries)
            return [topic] + queries
        logger.warning(
            "Query expansion attempt %d/%d: no usable queries in response. Raw: %s",
            attempt, total_attempts, raw_text[:300],
        )

    logger.warning(
        "Query expansion produced nothing usable for '%s' after %d attempt(s) — "
        "searching the topic string alone.", topic, total_attempts,
    )
    return [topic]

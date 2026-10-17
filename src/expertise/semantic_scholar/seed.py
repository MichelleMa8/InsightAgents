"""
Seed paper selection: search Semantic Scholar for recent papers on the topic
(widening the age window through S2_SEED_MAX_AGE_DAYS_STEPS until something is
found) and let the LLM pick one, or take a given paper id as the seed and derive
its topic from it when none is given.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional

import config as cfg
from expertise.semantic_scholar.client import S2Client
from expertise.semantic_scholar.paper_graph import render_api_papers
from json_llm import complete_json
from llm_client import LLMClient
from prompts.paper_graph_prompts import (
    SEED_SELECT_SYSTEM,
    SEED_SELECT_USER_TEMPLATE,
    TOPIC_FROM_PAPER_SYSTEM,
    TOPIC_FROM_PAPER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def reference_date() -> date:
    """"Now" for the recency window: today, or the pinned S2_REFERENCE_DATE."""
    if cfg.S2_REFERENCE_DATE:
        return date.fromisoformat(cfg.S2_REFERENCE_DATE)
    return date.today()


def _date_filter(max_age_days: Optional[int]) -> Optional[str]:
    """The API's publicationDateOrYear range, e.g. "2025-09-15:" — open-ended on
    the right, so "published on or after that date"."""
    if max_age_days is None:
        return None
    return f"{(reference_date() - timedelta(days=max_age_days)).isoformat()}:"


def _usable(paper: dict) -> bool:
    return bool(paper.get("paperId") and paper.get("title"))


def search_seed_candidates(
    topic: str, client: S2Client, limit: Optional[int] = None,
) -> List[dict]:
    """Relevance-ranked, recency-filtered papers for a topic, newest window first.

    Candidates with an abstract are moved ahead of ones without, keeping the
    API's relevance order within each group: the abstract is what the filtering
    prompt shows as the source paper on every expansion round, and a seed with
    no abstract weakens every round grown from it.

    `limit` overrides how many results are fetched per window (default
    S2_SEARCH_LIMIT).
    """
    for max_age_days in cfg.S2_SEED_MAX_AGE_DAYS_STEPS:
        window = _date_filter(max_age_days)
        results = [p for p in client.search_papers(
            topic, limit=limit or cfg.S2_SEARCH_LIMIT, publication_date_or_year=window,
        ) if _usable(p)]
        if not results:
            logger.info("No search results for '%s' in window %s — widening.",
                        topic, window or "any date")
            continue
        with_abstract = [p for p in results if p.get("abstract")]
        without = [p for p in results if not p.get("abstract")]
        ranked = with_abstract + without
        logger.info(
            "Seed search for '%s' (window %s): %d candidate(s), top = %s (%s)",
            topic, window or "any date", len(ranked),
            ranked[0].get("title", "?")[:80], ranked[0].get("publicationDate") or ranked[0].get("year"),
        )
        return ranked
    return []


def choose_seed(topic: str, candidates: List[dict], llm_client: LLMClient,
                stage: str = "s2_seed_select") -> List[dict]:
    """Reorder search results so the LLM's pick comes first.

    Search ranks by word overlap, which picks the wrong paper whenever the
    topic's discriminating term is rarer than its common ones — and since the
    whole graph grows from the seed, that single mistake decides the run. So the
    top S2_SEED_SHORTLIST
    results are shown to the LLM with their abstracts, and it names the one the
    graph should be grown from.

    The pick is moved to the front rather than replacing the list: the remaining
    candidates keep their relevance order and stay available as structural
    fallbacks. An unparseable or out-of-range answer leaves the order untouched,
    so this can only improve on the search ranking, never fail the topic.
    """
    shortlist = candidates[:cfg.S2_SEED_SHORTLIST]
    if len(shortlist) < 2:
        return candidates

    user_msg = SEED_SELECT_USER_TEMPLATE.format(
        topic=topic,
        candidates=render_api_papers(shortlist, cfg.S2_FILTER_ABSTRACT_CHARS),
    )
    try:
        parsed = complete_json(
            llm_client, SEED_SELECT_SYSTEM, user_msg,
            max_tokens=cfg.S2_SEED_SELECT_MAX_TOKENS, stage=stage,
            max_retries=cfg.S2_SEED_SELECT_MAX_RETRIES,
        )
    except RuntimeError as e:
        logger.warning("Seed selection failed (%s) — keeping the search ranking.", e)
        return candidates

    pick = parsed.get("pick")
    if not isinstance(pick, int) or not (1 <= pick <= len(shortlist)):
        logger.warning("Seed selection returned %r, which is not one of the %d candidates — "
                       "keeping the search ranking.", pick, len(shortlist))
        return candidates

    chosen = shortlist[pick - 1]
    logger.info("Seed pick: #%d %s — %s", pick, (chosen.get("title") or "?")[:80],
                parsed.get("why", ""))
    return [chosen] + [c for c in candidates if c is not chosen]


def fetch_seed_paper(paper_id: str, client: S2Client) -> dict:
    """The seed paper named directly by a seed-paper input file."""
    paper = client.get_paper(paper_id)
    if not paper or not _usable(paper):
        raise RuntimeError(f"Semantic Scholar does not know paper id '{paper_id}'")
    return paper


def topic_from_paper(paper: dict, llm_client: LLMClient) -> str:
    """Name the research topic of a seed paper, for seed-paper input that gives
    no topic of its own. The topic string drives every filtering call and the
    debate framing, so it is derived once and then checkpointed like any other
    pretrain result."""
    paper_text = (
        f"Title: {paper.get('title', '')}\n"
        f"Year: {paper.get('year', 'unknown')}\n"
        f"Abstract: {paper.get('abstract') or '(no abstract available)'}"
    )
    parsed = complete_json(
        llm_client, TOPIC_FROM_PAPER_SYSTEM,
        TOPIC_FROM_PAPER_USER_TEMPLATE.format(paper=paper_text),
        max_tokens=cfg.S2_TOPIC_FROM_PAPER_MAX_TOKENS,
        stage="s2_topic_from_paper",
        max_retries=cfg.INSIGHT_JSON_MAX_RETRIES,
    )
    topic = (parsed.get("topic") or "").strip()
    if not topic:
        raise RuntimeError("s2_topic_from_paper: response carried no topic string")
    logger.info("Derived topic from seed paper: '%s'", topic)
    return topic

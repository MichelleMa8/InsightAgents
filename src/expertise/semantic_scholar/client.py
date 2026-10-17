"""
Rate-limited Semantic Scholar API client for paper search and recommendations.
All threads share one pacing clock, 429/5xx responses are retried with
exponential backoff, and successful responses are cached on disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

import config as cfg

logger = logging.getLogger(__name__)

# Fields requested for every paper, everywhere — one shape for the whole
# pipeline, so a paper record from search and one from recommendations are
# interchangeable.
PAPER_FIELDS = (
    "paperId,title,abstract,year,publicationDate,citationCount,"
    "venue,authors,externalIds,url"
)

# One clock for the whole process: the 1 RPS quota is per API key, and every
# thread here shares that key.
_PACE_LOCK = threading.Lock()
_next_allowed_at = 0.0


class S2Error(RuntimeError):
    """A Semantic Scholar request that could not be completed after retries."""


def _pace() -> None:
    """Block until this process is allowed to issue the next S2 request."""
    global _next_allowed_at
    with _PACE_LOCK:
        now = time.monotonic()
        wait = _next_allowed_at - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _next_allowed_at = now + cfg.S2_MIN_REQUEST_INTERVAL


def _cache_path(path: str, params: Dict[str, Any]) -> Path:
    key = json.dumps([path, sorted(params.items())], sort_keys=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return Path(cfg.S2_CACHE_DIR) / f"{digest}.json"


class S2Client:
    """Thin wrapper over the two endpoints, with pacing, backoff and caching."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else cfg.S2_API_KEY
        if not self.api_key:
            logger.warning(
                "S2_API_KEY is not set — falling back to the shared unauthenticated "
                "pool, which is throttled hard on /paper/search. Expect slow runs."
            )
        if cfg.S2_CACHE_ENABLED:
            Path(cfg.S2_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[dict]:
        """GET one endpoint. Returns the decoded body, or None for a 404 (an id
        the API does not know). Raises S2Error when the retry budget is spent."""
        params = {k: v for k, v in params.items() if v is not None}
        cache_file = _cache_path(path, params) if cfg.S2_CACHE_ENABLED else None
        if cache_file is not None and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring unreadable S2 cache entry %s", cache_file)

        url = f"{cfg.S2_API_BASE}{path}"
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        last_error = "no attempt made"

        for attempt in range(cfg.S2_MAX_RETRIES + 1):
            _pace()
            try:
                with httpx.Client(timeout=cfg.S2_TIMEOUT) as client:
                    resp = client.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:                     # timeout, connection reset, …
                last_error = f"{type(e).__name__}: {e}"
                self._backoff(attempt, path, last_error)
                continue

            if resp.status_code == 200:
                body = resp.json()
                if cache_file is not None:
                    tmp = cache_file.with_suffix(".tmp")
                    tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                    tmp.replace(cache_file)
                return body
            if resp.status_code == 404:
                logger.info("S2 %s: 404 for params %s", path, params)
                return None
            # 429 and 5xx both mean "you are being throttled"; anything else is a
            # request-level problem that retrying cannot fix.
            if resp.status_code != 429 and resp.status_code < 500:
                raise S2Error(f"S2 {path} returned {resp.status_code}: {resp.text[:300]}")
            last_error = f"HTTP {resp.status_code}"
            self._backoff(attempt, path, last_error, retry_after=resp.headers.get("Retry-After"))

        raise S2Error(
            f"S2 {path} still failing after {cfg.S2_MAX_RETRIES + 1} attempt(s): {last_error}"
        )

    @staticmethod
    def _backoff(attempt: int, path: str, reason: str, retry_after: Optional[str] = None) -> None:
        """Exponential backoff with jitter. Retry-After is honoured when present,
        although the API is not currently observed to send it."""
        delay = min(cfg.S2_BACKOFF_MAX, cfg.S2_BACKOFF_BASE * (2 ** attempt))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        delay += random.uniform(0, 0.3 * delay)
        logger.info("S2 %s throttled (%s) — retry %d in %.1fs", path, reason, attempt + 1, delay)
        time.sleep(delay)

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def search_papers(
        self,
        query: str,
        limit: int,
        publication_date_or_year: Optional[str] = None,
    ) -> List[dict]:
        """Relevance-ranked search. `publication_date_or_year` is the API's own
        range filter, e.g. "2025-09-15:" for "published on or after that date"."""
        body = self._get(
            "/graph/v1/paper/search",
            {
                "query": query,
                "fields": PAPER_FIELDS,
                "limit": limit,
                "publicationDateOrYear": publication_date_or_year,
            },
        )
        return (body or {}).get("data", []) or []

    def recommendations(self, paper_id: str, limit: int, pool: Optional[str] = None) -> List[dict]:
        """Papers Semantic Scholar recommends for one paper."""
        body = self._get(
            f"/recommendations/v1/papers/forpaper/{paper_id}",
            {"fields": PAPER_FIELDS, "limit": limit, "from": pool},
        )
        return (body or {}).get("recommendedPapers", []) or []

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """One paper by id. Accepts every id form the API accepts, e.g.
        "649def34f8be52c8b66281af98ae884c09aef38b", "DOI:10.18653/v1/N18-3011",
        "ARXIV:2106.15928", "CorpusId:215416146", "URL:https://arxiv.org/abs/…"."""
        return self._get(f"/graph/v1/paper/{paper_id}", {"fields": PAPER_FIELDS})

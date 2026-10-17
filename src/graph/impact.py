"""
Forward-reference impact: number of other contributions that cite a given contribution.

forward_references.json (~348 MB) is loaded once and kept as a module-level singleton.
"""

import json
import logging
from functools import lru_cache
from typing import Dict, List

from config import DATA_DIR

logger = logging.getLogger(__name__)

_C2C: Dict[str, List[str]] = {}   # contribution_id -> [citing contribution_ids]
_C2P: Dict[str, List[str]] = {}   # contribution_id -> [citing paper corpus_ids]
_loaded = False


def _ensure_loaded() -> None:
    global _C2C, _C2P, _loaded
    if _loaded:
        return
    fwd_path = DATA_DIR / "metadata" / "forward_references.json"
    logger.info("Loading forward_references.json (~348 MB) …")
    with open(fwd_path, encoding="utf-8") as f:
        data = json.load(f)
    _C2C = data["forward_references_contribution_to_contribution"]
    _C2P = data["forward_references_contribution_to_paper"]
    _loaded = True
    logger.info("forward_references.json loaded: %d contribution entries", len(_C2C))


def preload() -> None:
    """Load the singleton now instead of on first use. Called before starting
    worker threads: _ensure_loaded() is unlocked, so N threads racing a cold
    cache would each parse the 348 MB file."""
    _ensure_loaded()


def get_impact(contribution_id: str) -> int:
    """Return the number of contributions that cite this contribution."""
    _ensure_loaded()
    return len(_C2C.get(contribution_id, []))


@lru_cache(maxsize=4096)
def get_paper_impact(corpus_id: str) -> int:
    """Return the number of distinct papers citing any contribution in this paper."""
    _ensure_loaded()
    paper_path = DATA_DIR / "papers" / str(corpus_id)[:2] / f"{corpus_id}.json"
    if not paper_path.exists():
        return 0

    with open(paper_path, encoding="utf-8") as f:
        paper = json.load(f)

    citing_papers = set()
    for contrib in paper.get("contributions", []):
        contribution_id = contrib.get("contribution_id")
        if contribution_id:
            citing_papers.update(str(pid) for pid in _C2P.get(contribution_id, []))

    citing_papers.discard(str(corpus_id))
    return len(citing_papers)


def get_forward_refs(contribution_id: str) -> List[str]:
    """Return the list of contribution_ids that cite this contribution."""
    _ensure_loaded()
    return _C2C.get(contribution_id, [])


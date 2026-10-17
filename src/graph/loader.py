"""Paper and contribution loading with in-process caches."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from config import DATA_DIR


def _paper_path(corpus_id: str) -> Path:
    return DATA_DIR / "papers" / corpus_id[:2] / f"{corpus_id}.json"


@lru_cache(maxsize=1)
def _load_manifest_metadata() -> dict[str, dict[str, Any]]:
    """Load compact paper metadata from the manifest for missing-field repair."""
    path = DATA_DIR / "metadata" / "paper_manifest.jsonl"
    metadata: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return metadata

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            corpus_id = str(row.get("corpus_id", ""))
            if not corpus_id:
                continue

            title = row.get("title")
            year = row.get("year")
            if title or year is not None:
                metadata[corpus_id] = {"title": title, "year": year}
    return metadata


def preload() -> None:
    """Load the module-level caches now instead of on first use — see
    graph.impact.preload for why callers want this before spawning threads."""
    _load_manifest_metadata()


def _repair_missing_metadata(paper: dict) -> dict:
    """Fill absent title/year from manifest metadata."""
    if paper.get("title") and paper.get("year") is not None:
        return paper

    corpus_id = str(paper.get("corpus_id", ""))
    if not corpus_id:
        return paper

    source = _load_manifest_metadata().get(corpus_id, {})
    title = source.get("title")
    year = source.get("year")

    if title and not paper.get("title"):
        paper["title"] = title
    if year is not None and paper.get("year") is None:
        paper["year"] = year
        publication_date = paper.get("publication_date")
        if isinstance(publication_date, dict) and publication_date.get("year") is None:
            publication_date["year"] = year
    return paper


@lru_cache(maxsize=4096)
def load_paper(corpus_id: str) -> dict:
    """Load a paper JSON by corpus_id. Results are cached in memory."""
    path = _paper_path(corpus_id)
    if not path.exists():
        raise FileNotFoundError(f"Paper not found: {corpus_id} (expected {path})")
    with open(path, encoding="utf-8") as f:
        return _repair_missing_metadata(json.load(f))


def corpus_id_from_contribution(contribution_id: str) -> str:
    """Extract the corpus_id from a contribution_id like '267751221.c0'."""
    return contribution_id.split(".")[0]


def find_contribution(paper: dict, contribution_id: str) -> Optional[dict]:
    """Return the contribution dict or None if not found."""
    for c in paper["contributions"]:
        if c["contribution_id"] == contribution_id:
            return c
    return None


def papers_have_citation_relation(paper_a: dict, paper_b: dict) -> bool:
    """
    Return True if paper_a references paper_b or paper_b references paper_a.
    A 'reference' means a corpus_id appears in at least one prerequisite reference.
    """
    id_a = str(paper_a["corpus_id"])
    id_b = str(paper_b["corpus_id"])

    def _cites(paper: dict, target_id: str) -> bool:
        for contrib in paper["contributions"]:
            for prereq in contrib["prerequisites"]:
                for ref in prereq["references"]:
                    if str(ref.get("corpus_id", "")) == target_id:
                        return True
        return False

    return _cites(paper_a, id_b) or _cites(paper_b, id_a)

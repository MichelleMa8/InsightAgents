"""
Resolves paper abstracts offline from the ACL Anthology BibTeX export, which is
downloaded on first use and cached as a title -> abstract index. Delete
ACL_CACHE_DIR (or call `build_index(force=True)`) to refresh it.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import urllib.request
from typing import Dict, Optional

from config import ACL_CACHE_DIR

logger = logging.getLogger(__name__)

BULK_URL   = "https://aclanthology.org/anthology+abstracts.bib.gz"
RAW_PATH   = ACL_CACHE_DIR / "anthology_abstracts.bib.gz"
INDEX_PATH = ACL_CACHE_DIR / "title_to_abstract.json"

_ENTRY_SPLIT_RE = re.compile(r"\n(?=@\w+\{)")
_TITLE_RE       = re.compile(r'title\s*=\s*"(.*?)",?\s*\n')
_ABSTRACT_RE    = re.compile(r'abstract\s*=\s*"(.*?)",?\s*\n')
_ACCENT_RE      = re.compile(r"\\[`'^\"~=.uvHtcdb]\s*(\w)")
_COMMAND_RE     = re.compile(r"\\\w+")

_index: Optional[Dict[str, str]] = None


def _normalize_title(title: str) -> str:
    """lowercase, collapse whitespace, decode LaTeX braces/accent escapes."""
    t = title.replace("{", "").replace("}", "")
    t = _ACCENT_RE.sub(r"\1", t)     # \'e, \"o, \~n, \c c, ... -> bare letter
    t = _COMMAND_RE.sub("", t)       # any remaining \command
    t = t.replace("\\", "")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _download_bulk_bibtex() -> None:
    ACL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ACL Anthology bulk BibTeX export from %s …", BULK_URL)
    urllib.request.urlretrieve(BULK_URL, RAW_PATH)
    logger.info("Downloaded %s (%d bytes).", RAW_PATH, RAW_PATH.stat().st_size)


def _parse_bulk_bibtex() -> Dict[str, str]:
    with gzip.open(RAW_PATH, "rt", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    index: Dict[str, str] = {}
    for entry in _ENTRY_SPLIT_RE.split(data):
        title_m = _TITLE_RE.search(entry)
        if not title_m:
            continue
        abstract_m = _ABSTRACT_RE.search(entry)
        if not abstract_m:
            continue
        norm_title = _normalize_title(title_m.group(1))
        if norm_title and norm_title not in index:
            index[norm_title] = abstract_m.group(1)

    logger.info("Parsed %d titles with abstracts from bulk BibTeX export.", len(index))
    return index


def build_index(force: bool = False) -> Dict[str, str]:
    """Build (or load cached) normalized_title -> abstract index."""
    global _index
    if _index is not None and not force:
        return _index

    if not force and INDEX_PATH.exists():
        with open(INDEX_PATH, encoding="utf-8") as f:
            _index = json.load(f)
        return _index

    if force or not RAW_PATH.exists():
        _download_bulk_bibtex()

    _index = _parse_bulk_bibtex()
    ACL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(_index, f, ensure_ascii=False)
    return _index


def resolve_abstract(title: Optional[str]) -> Optional[str]:
    """Return the ACL abstract for this paper title, or None if not indexed."""
    if not title:
        return None
    index = build_index()
    return index.get(_normalize_title(title))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index(force=True)

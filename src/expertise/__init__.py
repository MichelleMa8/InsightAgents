"""
Expertise backends that build the expert system prompts (Steps 1-3):
"semantic_scholar" (online, expertise/semantic_scholar/) and "contribution"
(offline, expertise/contribution.py). Both expose `run_pretrain(spec,
insight_client, progress, save, record_error)` and `result_section(progress)`,
and are imported lazily so an online run never loads the offline corpus.
"""

from __future__ import annotations

from types import ModuleType

import config as cfg

SOURCES = ("contribution", "semantic_scholar")
ALIASES = {"offline": "contribution", "online": "semantic_scholar"}


def resolve_source(source: str | None = None) -> str:
    """Canonical source name; accepts the "online" / "offline" aliases."""
    source = source or cfg.EXPERTISE_SOURCE
    return ALIASES.get(source, source)


def get_backend(source: str | None = None) -> ModuleType:
    """The backend module for `source` (default: config.EXPERTISE_SOURCE)."""
    source = resolve_source(source)
    if source == "contribution":
        from expertise import contribution
        return contribution
    if source == "semantic_scholar":
        from expertise import semantic_scholar
        return semantic_scholar
    raise ValueError(f"Unknown EXPERTISE_SOURCE {source!r} — expected one of {SOURCES}")

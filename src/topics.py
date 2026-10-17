"""
Loads the run's inputs into TopicSpecs: a topics file (a list of strings or
{"id", "topic"} objects) or, on the semantic_scholar source, a seed-papers file
of paper ids or {"id", "paper", "topic"} objects. A seed paper without a topic
gets one derived by the LLM later in the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

ROOT = Path(__file__).parent.parent

DEFAULT_TOPICS_FILE = ROOT / "data" / "topics.json"


@dataclass(frozen=True)
class TopicSpec:
    """One unit of work: a topic string, a seed paper, or both.

    `topic` is None only for a seed-paper entry that carries no topic of its
    own — the Semantic Scholar backend derives one from the paper and writes it
    back into the checkpoint. Every later step (filtering, insight, debate)
    always sees a real topic string.
    """

    topic: Optional[str] = None
    paper_id: Optional[str] = None
    entry_id: Optional[str] = None      # stable id from the input file, if any

    def __post_init__(self) -> None:
        if not self.topic and not self.paper_id:
            raise ValueError("a TopicSpec needs a topic, a paper id, or both")

    @property
    def display(self) -> str:
        """What this unit of work is called in logs before its topic is known."""
        return self.topic or f"paper:{self.paper_id}"

    @property
    def slug(self) -> str:
        """Output directory name. Derived from the topic when there is one; a
        seed-paper entry with no topic uses the paper id, because the derived
        topic is not known until the run is already writing into its directory."""
        if self.topic:
            return slugify(self.topic)
        return "paper_" + re.sub(r"[^a-z0-9]+", "_", (self.paper_id or "").lower()).strip("_")


def slugify(topic: str) -> str:
    return topic.lower().replace(" ", "_").replace("-", "_")


def _load_array(path: Union[str, Path]) -> list:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Input file {path} is not valid JSON: {e}") from None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Input file {path} must hold a non-empty JSON array")
    return raw


def load_topic_specs(path: Optional[Union[str, Path]] = None) -> List[TopicSpec]:
    """Read a topics file into TopicSpecs, in file order."""
    return [TopicSpec(topic=t) for t in load_topics(path)]


def load_seed_paper_specs(path: Union[str, Path]) -> List[TopicSpec]:
    """Read a seed-paper file into TopicSpecs, in file order.

    Paper ids are passed to Semantic Scholar as given, so every id form the API
    accepts works: a bare 40-hex paperId, "DOI:...", "ARXIV:...",
    "CorpusId:...", "URL:...".
    """
    specs: List[TopicSpec] = []
    for i, entry in enumerate(_load_array(path)):
        if isinstance(entry, str):
            paper_id, topic, entry_id = entry.strip(), None, None
        elif isinstance(entry, dict):
            paper_id = (entry.get("paper") or entry.get("paper_id") or "").strip()
            topic = (entry.get("topic") or "").strip() or None
            entry_id = entry.get("id")
        else:
            raise ValueError(f"Seed-paper file {path}: entry {i} is not a string or object")
        if not paper_id:
            raise ValueError(
                f"Seed-paper file {path}: entry {i} carries no paper id "
                f"(expected a string, or {{'paper': ...}})"
            )
        specs.append(TopicSpec(topic=topic, paper_id=paper_id, entry_id=entry_id))
    return specs


def load_topics(path: Optional[Union[str, Path]] = None) -> List[str]:
    """Read a topics file and return its topic strings, in file order."""
    path = Path(path) if path is not None else DEFAULT_TOPICS_FILE

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Topics file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Topics file {path} is not valid JSON: {e}") from None

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Topics file {path} must hold a non-empty JSON array")

    topics: List[str] = []
    for i, entry in enumerate(raw):
        topic = entry if isinstance(entry, str) else (
            entry.get("topic") if isinstance(entry, dict) else None
        )
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(
                f"Topics file {path}: entry {i} is not a topic string or {{'topic': ...}}"
            )
        topics.append(topic.strip())

    return topics

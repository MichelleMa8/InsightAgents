"""
The PaperGraph shared by both expertise paths: papers as nodes, older -> newer
edges (recommendation or citation links). Nodes carry short labels (P1, P2, …)
that prompts use instead of raw paper ids.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class PaperNode:
    paper_id: str
    label: str                 # graph-local handle: "P1", "P2", …
    title: str
    abstract: str
    year: Optional[int]
    publication_date: Optional[str]   # "YYYY-MM-DD" when the API knows it
    citation_count: Optional[int]
    venue: str
    authors: List[str]
    url: str
    level: int                 # 0 = seed, 1..len(*_KEEP_PER_LEVEL) = expansion round

    @classmethod
    def from_api(cls, raw: dict, label: str, level: int) -> "PaperNode":
        return cls(
            paper_id=raw.get("paperId") or "",
            label=label,
            title=raw.get("title") or "",
            abstract=raw.get("abstract") or "",
            year=raw.get("year"),
            publication_date=raw.get("publicationDate"),
            citation_count=raw.get("citationCount"),
            venue=raw.get("venue") or "",
            authors=[a.get("name", "") for a in (raw.get("authors") or [])],
            url=raw.get("url") or "",
            level=level,
        )

    def sort_key(self) -> str:
        """Publication date if known, else the year, else empty (sorts first)."""
        return self.publication_date or (f"{self.year}-01-01" if self.year else "")


@dataclass
class PaperEdge:
    edge_id: str               # "P1→P3", always older → newer
    from_label: str
    to_label: str
    from_paper_id: str
    to_paper_id: str
    expanded_from: str         # label of the paper whose expansion produced this edge
    level: int                 # expansion round that created it
    dated: bool = True         # False when no usable date existed on both sides and the
                               # expansion direction (parent → recommendation) was used instead
    relation: str = ""         # what the newer paper took from the older one — set on the
                               # contribution path (citation edges carry it), empty on S2


@dataclass
class PaperGraph:
    topic: str
    seed_paper_id: str
    nodes: List[PaperNode] = field(default_factory=list)
    edges: List[PaperEdge] = field(default_factory=list)

    # ── lookups ───────────────────────────────────────────────────────────────

    def by_label(self) -> Dict[str, PaperNode]:
        return {n.label: n for n in self.nodes}

    def by_paper_id(self) -> Dict[str, PaperNode]:
        return {n.paper_id: n for n in self.nodes}

    def years(self) -> List[int]:
        return sorted(n.year for n in self.nodes if n.year)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "seed_paper_id": self.seed_paper_id,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperGraph":
        return cls(
            topic=d["topic"],
            seed_paper_id=d["seed_paper_id"],
            nodes=[PaperNode(**n) for n in d["nodes"]],
            edges=[PaperEdge(**e) for e in d["edges"]],
        )

    def result_projection(self) -> dict:
        """The slim shape written into result.json — full text bodies dropped."""
        return {
            "topic": self.topic,
            "seed_paper_id": self.seed_paper_id,
            "nodes": [
                {
                    "label": n.label, "paper_id": n.paper_id, "title": n.title,
                    "year": n.year, "publication_date": n.publication_date,
                    "citation_count": n.citation_count, "venue": n.venue,
                    "level": n.level, "url": n.url,
                }
                for n in self.nodes
            ],
            "edges": [
                {"edge_id": e.edge_id, "from_label": e.from_label, "to_label": e.to_label,
                 "level": e.level, "dated": e.dated}
                for e in self.edges
            ],
        }


# ── Rendering for prompts ─────────────────────────────────────────────────────

def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_paper(node: PaperNode, abstract_chars: int) -> str:
    year = node.year or "year unknown"
    venue = f", {node.venue}" if node.venue else ""
    return (
        f"[{node.label}] {node.title} ({year}{venue}, {node.citation_count or 0} citations)\n"
        f"    {_truncate(node.abstract, abstract_chars) or '(no abstract available)'}"
    )


def render_api_papers(candidates: List[dict], abstract_chars: int) -> str:
    """Numbered list of raw Semantic Scholar records, for the prompts that choose
    among papers that are not in the graph yet (seed selection, recommendation
    filtering). Numbers rather than ids: a candidate number cannot be confused
    with a real paper, so a hallucinated pick is caught by a range check."""
    lines = []
    for i, c in enumerate(candidates, start=1):
        year = c.get("year") or "year unknown"
        lines.append(
            f"{i}. {c.get('title', '')} ({year}, {c.get('citationCount') or 0} citations)\n"
            f"   {_truncate(c.get('abstract') or '', abstract_chars) or '(no abstract available)'}"
        )
    return "\n".join(lines)


def render_nodes(graph: PaperGraph, abstract_chars: int,
                 round_header: str = "ROUND {level} (recommended from round {prev})") -> str:
    """Papers grouped by expansion round, oldest-first inside each round.
    `round_header` names how a round was reached ({level}, {prev})."""
    blocks: List[str] = []
    for level in sorted({n.level for n in graph.nodes}):
        header = "SEED PAPER" if level == 0 else round_header.format(level=level, prev=level - 1)
        rows = sorted((n for n in graph.nodes if n.level == level), key=PaperNode.sort_key)
        blocks.append(header + "\n" + "\n".join(render_paper(n, abstract_chars) for n in rows))
    return "\n\n".join(blocks)


def render_edges(graph: PaperGraph) -> str:
    """One line per edge, older paper first."""
    labels = graph.by_label()
    lines = []
    for e in sorted(graph.edges, key=lambda x: (x.level, x.edge_id)):
        src, dst = labels.get(e.from_label), labels.get(e.to_label)
        if not src or not dst:
            continue
        note = "" if e.dated else "  (no dates available — order is the expansion order)"
        if e.relation:
            note += f"\n      builds on: {_truncate(e.relation, 160)}"
        lines.append(
            f"  {e.edge_id}: {_truncate(src.title, 80)} ({src.year or '?'}) "
            f"→ {_truncate(dst.title, 80)} ({dst.year or '?'}){note}"
        )
    return "\n".join(lines)

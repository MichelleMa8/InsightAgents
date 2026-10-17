"""
Builds the offline PaperGraph: each round collects a paper's citation neighbours
(backward via prerequisites, forward via forward_references.json), keeps those
whose similarity to the topic clears CG_SIMILARITY_THRESHOLD, and lets the LLM
keep CG_KEEP_PER_LEVEL[i] of them. Edges run cited -> citing, and every citation
link among the final papers becomes an edge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import config as cfg
from expertise.semantic_scholar.paper_graph import PaperEdge, PaperGraph, PaperNode, render_paper
from graph.acl_abstracts import resolve_abstract
from graph.impact import get_forward_refs, get_impact, get_paper_impact
from graph.loader import corpus_id_from_contribution, find_contribution, load_paper
from json_llm import complete_json
from llm_client import LLMClient
from prompts.contribution_graph_prompts import CG_FILTER_SYSTEM, CG_FILTER_USER_TEMPLATE
from retrieval.searcher import get_contribution_embedding, retrieve_papers

logger = logging.getLogger(__name__)


# ── Local paper records ───────────────────────────────────────────────────────

def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _abstract(paper: dict) -> str:
    """The ACL Anthology abstract when indexed, else the paper's own
    contributions (name: description), which are what the corpus knows of it."""
    found = resolve_abstract(paper.get("title"))
    if found:
        return found
    parts = [f"{c.get('name', '')}: {c.get('description', '')}"
             for c in paper.get("contributions", [])]
    return _truncate(" ".join(parts), cfg.CG_ABSTRACT_CHARS)


def _publication_date(paper: dict) -> Optional[str]:
    d = paper.get("publication_date") or {}
    if not isinstance(d, dict) or not d.get("year"):
        return None
    return f"{d['year']:04d}-{d.get('month') or 1:02d}-{d.get('day') or 1:02d}"


def paper_record(corpus_id: str) -> Optional[dict]:
    """A local paper in the Semantic Scholar record shape, so the shared prompt
    renderers and PaperNode.from_api take it unchanged. None if not in the corpus."""
    try:
        paper = load_paper(str(corpus_id))
    except FileNotFoundError:
        return None
    return {
        "paperId": str(paper["corpus_id"]),
        "title": paper.get("title") or "",
        "abstract": _abstract(paper),
        "year": paper.get("year"),
        "publicationDate": _publication_date(paper),
        "citationCount": get_paper_impact(str(paper["corpus_id"])),
        "venue": "",
        "url": "",
    }


# ── Step 1: seed candidates ───────────────────────────────────────────────────

def _year_group(year: Optional[int]) -> int:
    for i, min_year in enumerate(cfg.CG_SEED_MIN_YEAR_STEPS):
        if min_year is None or (year or 0) >= min_year:
            return i
    return len(cfg.CG_SEED_MIN_YEAR_STEPS)


def search_seed_candidates(topic: str, query_vec: np.ndarray, queries: List[str]) -> List[dict]:
    """Retrieval-ranked seed candidates, newest year group first.

    Each record carries `retrieval_contribution`, the contribution it matched
    on (diagnostic only — the graph is grown from whole papers)."""
    hits = retrieve_papers(topic, top_k=cfg.CG_SEED_SEARCH_LIMIT,
                           query_vec=query_vec, queries=queries)
    records = []
    for hit in hits:
        rec = paper_record(hit.corpus_id)
        if rec and rec["title"]:
            rec["retrieval_contribution"] = hit.contribution_id
            records.append(rec)
    # Stable sort: relevance order is kept inside each year group.
    records.sort(key=lambda r: _year_group(r.get("year")))
    if records:
        logger.info("Seed search for '%s': %d candidate(s), top = %s (%s)", topic, len(records),
                    records[0]["title"][:80], records[0].get("year"))
    return records


# ── Citation neighbourhood ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _paper_years() -> Dict[str, int]:
    with open(cfg.DATA_DIR / "metadata" / "corpus_id_to_paper_year.json", encoding="utf-8") as f:
        return json.load(f)


def _anachronistic(cited_paper: str, citing_paper: str) -> bool:
    """A citing paper published before the paper it cites is an alignment
    artifact (e.g. a 2017 prerequisite matched onto a 2023 retrospective of the
    work it really cites), not a step in a line of work."""
    years = _paper_years()
    cited, citing = years.get(cited_paper), years.get(citing_paper)
    return bool(cited and citing and citing < cited)


@dataclass
class _Link:
    """All contribution-level citation links between one paper and one neighbour."""
    older_is_neighbour: bool                 # True: the paper builds on the neighbour
    far_contributions: Set[str] = field(default_factory=set)   # the neighbour's side
    pairs: Set[Tuple[str, str]] = field(default_factory=set)   # (cited cid, citing cid)


def _neighbours(corpus_id: str) -> Dict[str, _Link]:
    """neighbour corpus_id -> its links to this paper. Hub contributions on the
    far side (impact above TRAJ_HUB_IMPACT_CEILING) are skipped: citing BERT is
    not a step in anyone's line of work. So are links that run backwards in
    time (_anachronistic)."""
    try:
        paper = load_paper(corpus_id)
    except FileNotFoundError:
        return {}
    out: Dict[str, _Link] = {}

    def add(nid: str, older_is_neighbour: bool, far: str, pair: Tuple[str, str]) -> None:
        if nid == corpus_id or get_impact(far) > cfg.TRAJ_HUB_IMPACT_CEILING:
            return
        if _anachronistic(*((nid, corpus_id) if older_is_neighbour else (corpus_id, nid))):
            return
        link = out.setdefault(nid, _Link(older_is_neighbour))
        if link.older_is_neighbour != older_is_neighbour:
            return  # both directions (preprint quirks) — keep the first one seen
        link.far_contributions.add(far)
        link.pairs.add(pair)

    for contrib in paper.get("contributions", []):
        cid = contrib["contribution_id"]
        for prereq in contrib.get("prerequisites", []):
            for ref in prereq.get("references", []):
                for match in ref.get("matches", []):
                    mcid = match["contribution_id"]
                    add(corpus_id_from_contribution(mcid), True, mcid, (mcid, cid))
        for fcid in get_forward_refs(cid):
            add(corpus_id_from_contribution(fcid), False, fcid, (cid, fcid))
    return out


def _similarity(link: _Link, query_vec: np.ndarray) -> Optional[float]:
    sims = [float(np.dot(emb, query_vec)) for emb in
            (get_contribution_embedding(c) for c in link.far_contributions) if emb is not None]
    return max(sims) if sims else None


def _relation(pairs: Set[Tuple[str, str]]) -> str:
    """What the citing paper took from the cited one: the names of the citing
    contributions' prerequisites that matched, as the corpus states them."""
    names: List[str] = []
    for cited, citing in sorted(pairs):
        try:
            paper = load_paper(corpus_id_from_contribution(citing))
        except FileNotFoundError:
            continue
        contrib = find_contribution(paper, citing) or {}
        for prereq in contrib.get("prerequisites", []):
            if any(m.get("contribution_id") == cited
                   for ref in prereq.get("references", []) for m in ref.get("matches", [])):
                name = (prereq.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
    return "; ".join(names[:2])


def _make_edge(a: PaperNode, b: PaperNode, link: _Link, expanded_from: str, level: int) -> PaperEdge:
    """`link` is b's link as seen from a. Direction is the citation's: cited → citing."""
    older, newer = (b, a) if link.older_is_neighbour else (a, b)
    return PaperEdge(
        edge_id=f"{older.label}→{newer.label}",
        from_label=older.label, to_label=newer.label,
        from_paper_id=older.paper_id, to_paper_id=newer.paper_id,
        expanded_from=expanded_from, level=level, dated=True,
        relation=_relation(link.pairs),
    )


# ── Step 2: graph ─────────────────────────────────────────────────────────────

def _render_candidates(cands: List[Tuple[dict, _Link]]) -> str:
    lines = []
    for i, (rec, link) in enumerate(cands, start=1):
        direction = "the source paper builds on it" if link.older_is_neighbour else "it builds on the source paper"
        relation = _relation(link.pairs)
        lines.append(
            f"{i}. {rec['title']} ({rec.get('year') or 'year unknown'}, "
            f"{rec.get('citationCount') or 0} citations) — {direction}"
            + (f": {relation}" if relation else "") + "\n"
            f"   {_truncate(rec.get('abstract') or '', cfg.CG_ABSTRACT_CHARS) or '(no abstract available)'}"
        )
    return "\n".join(lines)


def _select_candidates(topic: str, parent: PaperNode, cands: List[Tuple[dict, _Link]],
                       keep_n: int, llm_client: LLMClient) -> List[Tuple[dict, _Link]]:
    """One filtering call over the parent's similarity-ranked neighbours not yet
    in the graph. An unparseable answer degrades to the similarity ranking."""
    user_msg = CG_FILTER_USER_TEMPLATE.format(
        topic=topic,
        source_paper=render_paper(parent, cfg.CG_ABSTRACT_CHARS),
        candidates=_render_candidates(cands),
        max_keep=keep_n,
    )
    try:
        parsed = complete_json(
            llm_client, CG_FILTER_SYSTEM.format(max_keep=keep_n), user_msg,
            max_tokens=cfg.CG_FILTER_MAX_TOKENS, stage="cg_filter",
            max_retries=cfg.CG_FILTER_MAX_RETRIES,
        )
    except RuntimeError as e:
        logger.warning("Filter call failed for %s (%s) — falling back to the top %d "
                       "neighbours by topic similarity.", parent.label, e, keep_n)
        return cands[:keep_n]

    selected, seen = [], set()
    for item in parsed.get("selected") or []:
        idx = item.get("id") if isinstance(item, dict) else item
        if not isinstance(idx, int) or not (1 <= idx <= len(cands)) or idx in seen:
            continue
        seen.add(idx)
        selected.append(cands[idx - 1])
        if len(selected) >= keep_n:
            break
    return selected


def _link_all(graph: PaperGraph) -> int:
    """Add every citation link between two papers of the finished graph that is
    not an edge yet. Returns how many were added."""
    by_id = graph.by_paper_id()
    have = {frozenset((e.from_label, e.to_label)) for e in graph.edges}
    added = 0
    for node in graph.nodes:
        for nid, link in _neighbours(node.paper_id).items():
            other = by_id.get(nid)
            if other is None or frozenset((node.label, other.label)) in have:
                continue
            have.add(frozenset((node.label, other.label)))
            graph.edges.append(_make_edge(node, other, link, node.label, max(node.level, other.level)))
            added += 1
    return added


def build_paper_graph(topic: str, seed_record: dict, query_vec: np.ndarray,
                      llm_client: LLMClient) -> PaperGraph:
    """Grow the graph from one seed paper. A parent with no usable neighbours is
    simply not expanded; validate_graph decides whether the result is usable."""
    seed = PaperNode.from_api(seed_record, label="P1", level=0)
    graph = PaperGraph(topic=topic, seed_paper_id=seed.paper_id, nodes=[seed], edges=[])
    nodes_by_id: Dict[str, PaperNode] = {seed.paper_id: seed}
    frontier: List[PaperNode] = [seed]
    next_label = 2

    for level, keep_n in enumerate(cfg.CG_KEEP_PER_LEVEL, start=1):
        if not frontier:
            logger.info("Round %d: nothing left to expand — stopping early.", level)
            break
        next_frontier: List[PaperNode] = []
        for parent in frontier:
            neighbours = _neighbours(parent.paper_id)
            scored: List[Tuple[float, str, _Link]] = []
            for nid, link in neighbours.items():
                if nid in nodes_by_id:
                    continue  # linked to the graph in the final sweep
                sim = _similarity(link, query_vec)
                if sim is not None and sim >= cfg.CG_SIMILARITY_THRESHOLD:
                    scored.append((sim, nid, link))
            scored.sort(key=lambda x: x[0], reverse=True)
            cands: List[Tuple[dict, _Link]] = []
            for _, nid, link in scored:
                rec = paper_record(nid)
                if rec and rec["title"]:
                    cands.append((rec, link))
                if len(cands) >= cfg.CG_CANDIDATES_PER_PAPER:
                    break
            if not cands:
                logger.info("Round %d: %s has no on-topic citation neighbours (%d raw).",
                            level, parent.label, len(neighbours))
                continue

            added = 0
            for rec, link in _select_candidates(topic, parent, cands, keep_n, llm_client):
                if rec["paperId"] in nodes_by_id:
                    continue  # picked by another parent earlier in this round
                node = PaperNode.from_api(rec, label=f"P{next_label}", level=level)
                next_label += 1
                nodes_by_id[node.paper_id] = node
                graph.nodes.append(node)
                next_frontier.append(node)
                graph.edges.append(_make_edge(parent, node, link, parent.label, level))
                added += 1
            logger.info("Round %d: %s → %d new paper(s) out of %d on-topic neighbour(s) (%d raw).",
                        level, parent.label, added, len(cands), len(neighbours))
        frontier = next_frontier

    cross = _link_all(graph)
    logger.info("Paper graph for '%s': %d papers, %d edges (%d cross-link(s)), %d round(s).",
                topic, len(graph.nodes), len(graph.edges), cross,
                max((n.level for n in graph.nodes), default=0))
    return graph


def validate_graph(graph: PaperGraph) -> Tuple[bool, Optional[str]]:
    """Size and depth as on the online path, plus the corpus recency ceiling:
    old papers are welcome, but the newest must reach SUBGRAPH_MIN_RECENT_YEAR
    or the insights describe a superseded era."""
    if len(graph.nodes) < cfg.CG_MIN_GRAPH_NODES:
        return False, f"only {len(graph.nodes)} papers (need {cfg.CG_MIN_GRAPH_NODES})"
    if not graph.edges:
        return False, "no edges"
    wanted = min(2, len(cfg.CG_KEEP_PER_LEVEL))
    depth = max(n.level for n in graph.nodes)
    if depth < wanted:
        return False, f"expansion stopped after round {depth} (needed {wanted})"
    if cfg.SUBGRAPH_MIN_RECENT_YEAR is not None:
        newest = max(graph.years(), default=None)
        if newest is None or newest < cfg.SUBGRAPH_MIN_RECENT_YEAR:
            return False, (f"newest paper is from {newest}, before "
                           f"SUBGRAPH_MIN_RECENT_YEAR={cfg.SUBGRAPH_MIN_RECENT_YEAR}")
    return True, None

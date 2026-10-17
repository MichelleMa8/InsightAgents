"""
Grows the online PaperGraph by recursively expanding Semantic Scholar
recommendations, with an LLM filter keeping S2_KEEP_PER_LEVEL[i] topic-relevant
papers per parent at each round. Recommendations that point at a paper already
in the graph are added as edges without asking the LLM.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import config as cfg
from expertise.semantic_scholar.client import S2Client, S2Error
from expertise.semantic_scholar.paper_graph import (
    PaperEdge,
    PaperGraph,
    PaperNode,
    render_api_papers,
    render_paper,
)
from json_llm import complete_json
from llm_client import LLMClient
from prompts.paper_graph_prompts import REC_FILTER_SYSTEM, REC_FILTER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def _usable(raw: dict) -> bool:
    return bool(raw.get("paperId") and raw.get("title"))


def _select_candidates(
    topic: str,
    parent: PaperNode,
    candidates: List[dict],
    keep_n: int,
    llm_client: LLMClient,
) -> List[dict]:
    """One filtering call, over candidates that are not in the graph yet.

    Papers already in the graph never reach this call: their edge to this parent
    is recorded structurally (the recommendation relation is the edge), so
    selecting them would add nothing and would only spend the quota that decides
    which *new* papers get in. Returns the selected records in the order the LLM
    picked them, capped at keep_n."""
    user_msg = REC_FILTER_USER_TEMPLATE.format(
        topic=topic,
        source_paper=render_paper(parent, cfg.S2_FILTER_ABSTRACT_CHARS),
        candidates=render_api_papers(candidates, cfg.S2_FILTER_ABSTRACT_CHARS),
        max_keep=keep_n,
    )
    try:
        parsed = complete_json(
            llm_client, REC_FILTER_SYSTEM.format(max_keep=keep_n), user_msg,
            max_tokens=cfg.S2_FILTER_MAX_TOKENS, stage="s2_rec_filter",
            max_retries=cfg.S2_FILTER_MAX_RETRIES,
        )
    except RuntimeError as e:
        # Dropping the whole branch would be a bigger distortion than taking the
        # API's own ranking, which is at least relevance-ordered — but it is a
        # degraded round and must be visible as one.
        logger.warning("Filter call failed for %s (%s) — falling back to the top %d "
                       "recommendations unfiltered.", parent.label, e, keep_n)
        return candidates[:keep_n]

    selected: List[dict] = []
    seen: Set[int] = set()
    for item in parsed.get("selected") or []:
        idx = item.get("id") if isinstance(item, dict) else item
        if not isinstance(idx, int) or not (1 <= idx <= len(candidates)) or idx in seen:
            logger.debug("Filter returned unusable selection %r for %s", item, parent.label)
            continue
        seen.add(idx)
        selected.append(candidates[idx - 1])
        if len(selected) >= keep_n:
            break
    return selected


def _make_edge(parent: PaperNode, child: PaperNode, level: int) -> PaperEdge:
    """Direction is temporal: older paper → newer paper. Recommendation is
    symmetric, publication dates are not, so the dates are what decide which
    paper is the prerequisite and which one builds on it. With no usable date on
    both sides, the expansion order (parent → recommendation) stands in, and the
    edge is flagged so nothing downstream reads it as a dated claim."""
    p_key, c_key = parent.sort_key(), child.sort_key()
    dated = bool(p_key and c_key)
    older, newer = (child, parent) if dated and c_key < p_key else (parent, child)
    return PaperEdge(
        edge_id=f"{older.label}→{newer.label}",
        from_label=older.label, to_label=newer.label,
        from_paper_id=older.paper_id, to_paper_id=newer.paper_id,
        expanded_from=parent.label, level=level, dated=dated,
    )


def build_paper_graph(
    topic: str,
    seed_raw: dict,
    client: S2Client,
    llm_client: LLMClient,
) -> PaperGraph:
    """Grow the graph from one seed paper. Never raises on API trouble for a
    single parent — that branch is simply not expanded — so a finished graph can
    still be smaller than the configured fan-out; validate_graph decides whether
    what came back is usable."""
    seed = PaperNode.from_api(seed_raw, label="P1", level=0)
    graph = PaperGraph(topic=topic, seed_paper_id=seed.paper_id, nodes=[seed], edges=[])
    nodes_by_id: Dict[str, PaperNode] = {seed.paper_id: seed}
    edge_ids: Set[str] = set()
    frontier: List[PaperNode] = [seed]
    next_label = 2

    for level, keep_n in enumerate(cfg.S2_KEEP_PER_LEVEL, start=1):
        if not frontier:
            logger.info("Round %d: nothing left to expand — stopping early.", level)
            break
        next_frontier: List[PaperNode] = []
        for parent in frontier:
            try:
                recs = client.recommendations(
                    parent.paper_id, limit=cfg.S2_RECS_PER_PAPER, pool=cfg.S2_REC_POOL,
                )
            except S2Error as e:
                logger.warning("No recommendations for %s (%s): %s", parent.label, parent.paper_id, e)
                continue
            candidates = [r for r in recs if _usable(r) and r["paperId"] != parent.paper_id]
            if not candidates:
                logger.info("Round %d: %s has no usable recommendations.", level, parent.label)
                continue

            # A recommendation pointing at a paper the graph already holds IS an
            # edge — the relation is a fact about the API's output, not a judgment
            # the filter has to confirm. Recording it here (rather than only when
            # the filter happens to pick it) is what turns the expansion tree into
            # a DAG: it links branches that arrived at the same work from different
            # directions. Note this only catches papers already in the graph *at
            # this moment*; a recommendation whose target joins the graph later,
            # through another branch, is not revisited.
            linked = 0
            fresh: List[dict] = []
            for cand in candidates:
                existing = nodes_by_id.get(cand["paperId"])
                if existing is None:
                    fresh.append(cand)
                    continue
                edge = _make_edge(parent, existing, level)
                if edge.edge_id not in edge_ids:
                    edge_ids.add(edge.edge_id)
                    graph.edges.append(edge)
                    linked += 1

            # The filter only ever decides which *new* papers join the graph.
            selected = _select_candidates(topic, parent, fresh, keep_n, llm_client) if fresh else []
            added = 0
            for cand in selected:
                if added >= keep_n:
                    break
                node = PaperNode.from_api(cand, label=f"P{next_label}", level=level)
                next_label += 1
                nodes_by_id[node.paper_id] = node
                graph.nodes.append(node)
                next_frontier.append(node)
                added += 1
                edge = _make_edge(parent, node, level)
                if edge.edge_id not in edge_ids:
                    edge_ids.add(edge.edge_id)
                    graph.edges.append(edge)
            logger.info(
                "Round %d: %s → %d new paper(s), %d cross-link(s) out of %d recommendation(s).",
                level, parent.label, added, linked, len(candidates),
            )
        frontier = next_frontier

    logger.info("Paper graph for '%s': %d papers, %d edges, %d round(s).",
                topic, len(graph.nodes), len(graph.edges),
                max((n.level for n in graph.nodes), default=0))
    return graph


def validate_graph(graph: PaperGraph) -> Tuple[bool, Optional[str]]:
    """A graph too small or too shallow cannot support several distinct
    insights, so the caller tries the next seed candidate instead."""
    if len(graph.nodes) < cfg.S2_MIN_GRAPH_NODES:
        return False, f"only {len(graph.nodes)} papers (need {cfg.S2_MIN_GRAPH_NODES})"
    if not graph.edges:
        return False, "no edges"
    # Two rounds is what makes this a graph of a research line rather than one
    # paper's neighbour list — but never demand more rounds than are configured.
    wanted = min(2, len(cfg.S2_KEEP_PER_LEVEL))
    depth = max(n.level for n in graph.nodes)
    if depth < wanted:
        return False, f"expansion stopped after round {depth} (needed {wanted})"
    return True, None

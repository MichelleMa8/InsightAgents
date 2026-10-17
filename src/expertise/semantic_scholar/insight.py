"""
Generates all S2_NUM_INSIGHTS insights from one paper graph in a single LLM
call, so they come out distinct, and fills each into an expert system prompt.
Each insight's evidence subset of papers and edges becomes that expert's
grounding.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import config as cfg
from expertise.semantic_scholar.paper_graph import PaperGraph, render_edges, render_nodes
from json_llm import complete_json
from llm_client import LLMClient
from prompts.paper_agent_prompts import (
    PAPER_AGENT_SYSTEM_PROMPT,
    PAPER_BLOCK_TEMPLATE,
)
from prompts.paper_graph_prompts import (
    GRAPH_INSIGHT_SYSTEM,
    GRAPH_INSIGHT_USER_TEMPLATE,
    GRAPH_PROVENANCE,
)

logger = logging.getLogger(__name__)

_TEXT_FIELDS = ("insight_name", "core_insight", "historical_tension", "grounding_summary")


def _norm_edge_id(edge_id: str) -> str:
    """"P3 -> P7" / "P3-->P7" / "p3→p7" all mean the same edge."""
    return re.sub(r"\s+", "", edge_id or "").replace("-->", "→").replace("->", "→").upper()


def _resolve_papers(raw_papers: Any, graph: PaperGraph) -> List[dict]:
    """Keep only entries naming a paper that is actually in the graph."""
    by_label = graph.by_label()
    out, seen = [], set()
    for entry in raw_papers or []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "")).strip().upper()
        node = by_label.get(label)
        if node is None or label in seen:
            if node is None:
                logger.warning("Insight cites unknown paper label %r — dropped.", entry.get("label"))
            continue
        seen.add(label)
        out.append({
            "label": node.label,
            "paper_id": node.paper_id,
            "title": node.title,
            "year": node.year,
            "neutral_description": str(entry.get("neutral_description", "")).strip(),
            "role_in_insight": str(entry.get("role_in_insight", "")).strip(),
        })
    return out


def _resolve_edges(raw_edges: Any, graph: PaperGraph) -> List[dict]:
    """Keep only entries naming an edge that is actually in the graph. An edge
    written in the wrong direction is corrected rather than dropped — the
    direction is a fact about publication dates, not about the insight."""
    by_id = {_norm_edge_id(e.edge_id): e for e in graph.edges}
    by_label = graph.by_label()
    out, seen = [], set()
    for entry in raw_edges or []:
        if not isinstance(entry, dict):
            continue
        key = _norm_edge_id(str(entry.get("edge_id", "")))
        edge = by_id.get(key)
        if edge is None and "→" in key:
            a, b = key.split("→", 1)
            edge = by_id.get(f"{b}→{a}")
        if edge is None or edge.edge_id in seen:
            if edge is None:
                logger.warning("Insight cites unknown edge %r — dropped.", entry.get("edge_id"))
            continue
        seen.add(edge.edge_id)
        src, dst = by_label.get(edge.from_label), by_label.get(edge.to_label)
        out.append({
            "edge_id": edge.edge_id,
            "from_label": edge.from_label,
            "to_label": edge.to_label,
            "from_year": src.year if src else None,
            "to_year": dst.year if dst else None,
            "transition_summary": str(entry.get("transition_summary", "")).strip(),
            "role_in_abstraction": str(entry.get("role_in_abstraction", "")).strip(),
        })
    return out


def _complete_evidence(insight: dict, graph: PaperGraph) -> None:
    """Make the evidence self-consistent: every paper named by a cited edge is
    also in the paper list. Without this an expert can be handed "edge P5→P16
    supports this" while holding no memory entry for P16 at all — a dangling
    reference, and exactly the kind of gap a debate fills by inventing."""
    by_label = graph.by_label()
    have = {p["label"] for p in insight["papers"]}
    for edge in insight["edges"]:
        for label in (edge["from_label"], edge["to_label"]):
            node = by_label.get(label)
            if node is None or label in have:
                continue
            have.add(label)
            insight["papers"].append({
                "label": node.label,
                "paper_id": node.paper_id,
                "title": node.title,
                "year": node.year,
                "neutral_description": node.title,
                "role_in_insight": "the other end of a step this insight cites",
            })


def _summary_edge_refs(insight: dict, graph: PaperGraph) -> List[str]:
    """Edge ids named in the prose fields that are not backed by the graph or by
    this insight's own edge list — they would reach the expert as evidence that
    cannot be checked."""
    cited = {e["edge_id"] for e in insight["edges"]}
    in_graph = {e.edge_id for e in graph.edges}
    prose = " ".join(insight.get(f, "") for f in _TEXT_FIELDS)
    bad = []
    for ref in re.findall(r"P\d+\s*(?:→|-+>)\s*P\d+", prose):
        norm = _norm_edge_id(ref)
        if norm not in cited and norm not in in_graph:
            bad.append(ref)
    return bad


def _resolve_claims(raw_claims, graph: PaperGraph) -> List[dict]:
    """Claims whose label names a paper actually in the graph, capped at 6.
    Each claim is one specific finding an expert can build on and cite; a claim
    citing a paper the graph does not hold would be unverifiable, so it is
    dropped rather than repaired."""
    by_label = graph.by_label()
    claims = []
    for c in raw_claims or []:
        if not isinstance(c, dict):
            continue
        label = str(c.get("label", "")).strip()
        finding = str(c.get("finding", "")).strip()
        if finding and label in by_label:
            claims.append({"label": label, "finding": finding})
    return claims[:6]


def _normalise(parsed: dict, graph: PaperGraph) -> Tuple[List[dict], List[str]]:
    """Resolve every insight's evidence against the graph and report what is
    wrong with the set as a whole (empty list = nothing to re-ask about)."""
    insights: List[dict] = []
    problems: List[str] = []

    for i, raw in enumerate(parsed.get("insights") or [], start=1):
        if not isinstance(raw, dict):
            continue
        insight = {f: str(raw.get(f, "")).strip() for f in _TEXT_FIELDS}
        insight["papers"] = _resolve_papers(raw.get("papers"), graph)
        insight["edges"] = _resolve_edges(raw.get("edges"), graph)
        insight["claims"] = _resolve_claims(raw.get("claims"), graph)
        if not insight["core_insight"]:
            problems.append(f"insight {i} has an empty core_insight")
            continue
        name = insight["insight_name"] or f"insight {i}"
        _complete_evidence(insight, graph)
        if len(insight["papers"]) < cfg.S2_INSIGHT_MIN_EVIDENCE_PAPERS:
            problems.append(
                f"'{name}' cites {len(insight['papers'])} paper(s) from the graph, need at "
                f"least {cfg.S2_INSIGHT_MIN_EVIDENCE_PAPERS}"
            )
            continue
        if len(insight["edges"]) < cfg.S2_INSIGHT_MIN_EVIDENCE_EDGES:
            problems.append(
                f"'{name}' cites {len(insight['edges'])} edge(s) that exist in the graph, "
                f"need at least {cfg.S2_INSIGHT_MIN_EVIDENCE_EDGES} — an insight is a pattern "
                f"across steps, so one step cannot establish it"
            )
            continue
        if len(insight["claims"]) < cfg.S2_INSIGHT_MIN_CLAIMS:
            problems.append(
                f"'{name}' has {len(insight['claims'])} claims backed by graph papers, "
                f"need at least {cfg.S2_INSIGHT_MIN_CLAIMS} — each claim states one "
                f"finding and cites its [P#] paper"
            )
            continue
        dangling = _summary_edge_refs(insight, graph)
        if dangling:
            problems.append(
                f"'{name}' names edge(s) {', '.join(dangling)} in its prose that are not in "
                f"the graph and not in its own edge list"
            )
            continue
        insights.append(insight)

    if len(insights) != cfg.S2_NUM_INSIGHTS:
        problems.append(
            f"{len(insights)} usable insight(s) came back, {cfg.S2_NUM_INSIGHTS} are required"
        )
    return insights, problems


def _log_overlap(insights: List[dict]) -> None:
    """The insights are supposed to rest on mostly different papers; when
    they do not, the experts will be near-duplicates of each other and that
    should be visible in the log rather than only in the transcript."""
    sets = [{p["label"] for p in ins["papers"]} for ins in insights]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            if len(shared) > 1:
                logger.warning(
                    "Insights %d and %d share %d papers (%s) — their experts may overlap.",
                    i + 1, j + 1, len(shared), ", ".join(sorted(shared)),
                )


def generate_insights(
    graph: PaperGraph, topic: str, llm_client: LLMClient,
    provenance: str = GRAPH_PROVENANCE,
    nodes_text: Optional[str] = None,
    stage: str = "s2_insight",
) -> List[dict]:
    """S2_NUM_INSIGHTS insights over the whole graph, each with a resolved
    evidence subset. One re-ask is spent when the first answer is short of
    usable insights (evidence not in the graph, or below the evidence/claim
    floors); after that the run
    continues with whatever is usable (a debate with fewer experts still runs)
    and fails only if nothing is.

    `provenance` / `nodes_text` let the contribution path substitute its
    citation-graph provenance and round headers."""
    base_user = GRAPH_INSIGHT_USER_TEMPLATE.format(
        topic=topic,
        nodes_text=nodes_text if nodes_text is not None else render_nodes(graph, cfg.S2_INSIGHT_ABSTRACT_CHARS),
        edges_text=render_edges(graph) or "  (none)",
        num_insights=cfg.S2_NUM_INSIGHTS,
    )
    system = GRAPH_INSIGHT_SYSTEM.format(
        num_insights=cfg.S2_NUM_INSIGHTS,
        min_papers=cfg.S2_INSIGHT_MIN_EVIDENCE_PAPERS,
        min_edges=cfg.S2_INSIGHT_MIN_EVIDENCE_EDGES,
        provenance=provenance,
    )

    best: List[dict] = []
    feedback = ""
    for attempt in (1, 2):
        parsed = complete_json(
            llm_client, system, base_user + feedback,
            max_tokens=cfg.S2_INSIGHT_MAX_TOKENS, stage=stage,
            max_retries=cfg.INSIGHT_JSON_MAX_RETRIES,
        )
        insights, problems = _normalise(parsed, graph)
        # A re-ask can come back worse than what it was asked to fix, and a
        # missing expert is the expensive failure here — so the better of the
        # two attempts is what the run keeps.
        if len(insights) > len(best):
            best = insights
        if not problems:
            break
        logger.warning("Insight attempt %d: %s", attempt, "; ".join(problems))
        if attempt == 2:
            break
        feedback = (
            "\n\nYour previous answer had these problems:\n- "
            + "\n- ".join(problems)
            + f"\n\nReturn {cfg.S2_NUM_INSIGHTS} insights again, fixing them. Use only the "
              "[P#] labels and edge ids listed above."
        )

    if not best:
        raise RuntimeError(f"{stage}: no usable insight came back from the graph")
    if len(best) < cfg.S2_NUM_INSIGHTS:
        logger.error("Only %d of %d insights are usable — the debate runs with %d expert(s).",
                     len(best), cfg.S2_NUM_INSIGHTS, len(best))
    _log_overlap(best)
    return best


# ── Expert system prompt ──────────────────────────────────────────────────────

def render_evidence_papers(insight: Dict[str, Any], graph: Optional[PaperGraph]) -> str:
    """The insight's own evidence papers as title/authors/year/venue/abstract
    blocks. Without a graph (or for a label the graph lacks) only the stored
    title and year are shown."""
    if not (insight.get("papers") or []):
        return ""
    by_label = graph.by_label() if graph is not None else {}
    blocks = []
    # Keyed by the paper's graph label — the same id the claims cite — so the
    # experts and the moderator's paper directory share one id space.
    for paper in insight.get("papers") or []:
        index = paper["label"]
        node = by_label.get(paper["label"])
        if node is None:
            blocks.append(PAPER_BLOCK_TEMPLATE.format(
                index=index, title=paper.get("title", ""), authors="unknown",
                year=paper.get("year") or "year unknown", venue="",
                abstract="(abstract not available)"))
            continue
        authors = ", ".join(node.authors[:6]) + (" et al." if len(node.authors) > 6 else "")
        blocks.append(PAPER_BLOCK_TEMPLATE.format(
            index=index, title=node.title, authors=authors or "unknown",
            year=node.year or "year unknown",
            venue=f", {node.venue}" if node.venue else "",
            abstract=" ".join((node.abstract or "").split()) or "(no abstract available)"))
    return ("\nThe papers that reading rests on:\n" + "".join(blocks)
            + "\nYou have abstracts, not full papers, so say when a detail is not in them.\n")


def render_claims(insight: Dict[str, Any]) -> str:
    """The insight's claims as a findings block for the expert prompt. Empty
    when the insight carries no claims."""
    claims = insight.get("claims") or []
    if not claims:
        return ""
    lines = "\n".join(f"- [{c['label']}] {c['finding']}" for c in claims)
    return ("\nFindings this reading rests on — specifics you can build on and "
            "cite by name:\n" + lines + "\n")


def fill_system_prompt(insight: Dict[str, Any], graph: Optional[PaperGraph] = None) -> str:
    """Fill PAPER_AGENT_SYSTEM_PROMPT from one insight.

    The prompt carries the insight, its claims, and the full abstracts of its
    own evidence papers — never the rest of the graph.
    """
    return PAPER_AGENT_SYSTEM_PROMPT.format(
        INSIGHT_NAME=insight.get("insight_name", ""),
        CORE_INSIGHT=insight.get("core_insight", ""),
        HISTORICAL_TENSION=insight.get("historical_tension", ""),
        CLAIMS_BLOCK=render_claims(insight),
        PAPER_BLOCKS=render_evidence_papers(insight, graph),
    )

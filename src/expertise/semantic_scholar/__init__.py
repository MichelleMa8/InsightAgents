"""
Online expertise backend (EXPERTISE_SOURCE = "semantic_scholar"): picks a recent
seed paper via the Semantic Scholar API, grows an LLM-filtered recommendation
graph around it, and distils S2_NUM_INSIGHTS insights from that graph, one per
expert.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import config as cfg
import cost_report
from expertise.semantic_scholar import builder, insight as insight_mod, seed as seed_mod
from expertise.semantic_scholar.client import S2Client
from expertise.semantic_scholar.paper_graph import PaperGraph
from llm_client import LLMClient

logger = logging.getLogger(__name__)

SOURCE_NAME = "semantic_scholar"

# Only the fields the pipeline itself needs are checkpointed, not the whole API
# record (author lists and openAccessPdf blocks are the bulk of it).
_SEED_FIELDS = ("paperId", "title", "abstract", "year", "publicationDate",
                "citationCount", "venue", "url")


def _trim(paper: dict) -> dict:
    return {k: paper.get(k) for k in _SEED_FIELDS}


def run_pretrain(
    spec,
    insight_client: LLMClient,
    progress: dict,
    save: Callable[[], None],
    record_error: Callable[[dict, Optional[str], str], None],
) -> None:
    """Steps 1-3 with per-step checkpointing into `progress`. `spec` is a
    topics.TopicSpec: it carries a topic, a seed paper id, or both."""
    if progress["insight"] and progress["insight"].get("done"):
        return

    client = S2Client()

    # ── 1. Seed paper ─────────────────────────────────────────────────────────
    if not progress["retrieval"]:
        logger.info("[1/3] Selecting seed paper …")
        call_snap = insight_client.call_log_snapshot()
        t0 = time.time()
        if spec.paper_id:
            candidates = [seed_mod.fetch_seed_paper(spec.paper_id, client)]
            topic = spec.topic or seed_mod.topic_from_paper(candidates[0], insight_client)
        else:
            topic = spec.topic
            hits = seed_mod.search_seed_candidates(topic, client)
            # Relevance ranking alone is not enough to pick a seed (see
            # seed.choose_seed); the LLM picks from the shortlist, and the rest
            # stay as structural fallbacks in their original order.
            candidates = seed_mod.choose_seed(topic, hits, insight_client)[:cfg.S2_SEED_MAX_ATTEMPTS]
        calls = insight_client.calls_since(call_snap)

        if not candidates:
            logger.error("No seed paper found for '%s'. Skipping.", spec.display)
            record_error(progress, "retrieval", "no seed paper found")
            save()
            return

        progress["topic"] = topic          # may have just been derived from the paper
        progress["retrieval"] = {
            "topic": topic,
            "seed_candidates": [_trim(p) for p in candidates],
            "next_candidate_idx": 0,
            "seed_paper_id": None,
        }
        progress["pretrain_stages"].append(cost_report.make_stage_entry(
            stage_id="retrieval",
            label="Step 1 — Seed paper (recent-only relevance search + LLM pick of the best fit)",
            calls=calls, duration_sec=time.time() - t0,
            model=insight_client.model if calls else None,
        ))
        save()

    topic = progress["retrieval"]["topic"]
    candidates = progress["retrieval"]["seed_candidates"]

    # ── 2. Paper graph ────────────────────────────────────────────────────────
    graph_state = progress["trajectory"]
    if graph_state is None:
        graph_state = progress["trajectory"] = {
            "done": False, "graph": None, "calls": [], "duration_sec": 0.0,
        }
    if not graph_state["done"]:
        logger.info("[2/3] Building paper graph (depth %d, fan-out %s) …",
                    len(cfg.S2_KEEP_PER_LEVEL), cfg.S2_KEEP_PER_LEVEL)
        while progress["retrieval"]["next_candidate_idx"] < len(candidates):
            idx = progress["retrieval"]["next_candidate_idx"]
            seed_paper = candidates[idx]
            call_snap = insight_client.call_log_snapshot()
            t0 = time.time()
            graph = builder.build_paper_graph(topic, seed_paper, client, insight_client)
            graph_state["calls"].extend(insight_client.calls_since(call_snap))
            graph_state["duration_sec"] += time.time() - t0
            ok, reason = builder.validate_graph(graph)
            progress["retrieval"]["next_candidate_idx"] = idx + 1
            if ok:
                progress["retrieval"]["seed_paper_id"] = graph.seed_paper_id
                graph_state["graph"] = graph.to_dict()
                graph_state["done"] = True
                save()
                break
            logger.warning(
                "Graph from seed candidate %d/%d ('%s') rejected: %s — trying the next seed.",
                idx + 1, len(candidates), (seed_paper.get("title") or "?")[:80], reason,
            )
            save()

        progress["pretrain_stages"].append(cost_report.make_stage_entry(
            stage_id="trajectory",
            label="Step 2 — Paper graph (recursive Semantic Scholar recommendations, LLM-filtered)",
            calls=graph_state["calls"], duration_sec=graph_state["duration_sec"],
            model=insight_client.model,
        ))
        save()
        if not graph_state["done"]:
            logger.error("No valid paper graph for '%s' (tried %d seed candidate(s)).",
                         topic, len(candidates))
            record_error(progress, "trajectory", "no valid paper graph")
            save()
            return

    graph = PaperGraph.from_dict(graph_state["graph"])

    # ── 3. Insights ───────────────────────────────────────────────────────────
    ins_state = progress["insight"]
    if ins_state is None:
        ins_state = progress["insight"] = {
            "done": False, "insights": [], "expert_prompts": [], "calls": [], "duration_sec": 0.0,
        }
    if not ins_state["done"]:
        logger.info("[3/3] Distilling %d insight(s) from %d papers …",
                    cfg.S2_NUM_INSIGHTS, len(graph.nodes))
        call_snap = insight_client.call_log_snapshot()
        t0 = time.time()
        insights = insight_mod.generate_insights(graph, topic, insight_client)
        prompts = [insight_mod.fill_system_prompt(ins, graph) for ins in insights]
        ins_state["calls"].extend(insight_client.calls_since(call_snap))
        ins_state["duration_sec"] += time.time() - t0
        ins_state["insights"] = insights
        ins_state["expert_prompts"] = prompts
        ins_state["done"] = True
        for i, ins in enumerate(insights):
            logger.info("  Expert %s insight: %s (evidence: %s)", chr(65 + i),
                        ins.get("insight_name", "?"),
                        ", ".join(p["label"] for p in ins["papers"]))
        progress["pretrain_stages"].append(cost_report.make_stage_entry(
            stage_id="insight",
            label=f"Step 3 — Insight generation ({cfg.S2_NUM_INSIGHTS} insights from one graph, 1 call)",
            calls=ins_state["calls"], duration_sec=ins_state["duration_sec"],
            model=insight_client.model,
        ))
        save()


def _evidence_projection(ins: dict) -> dict:
    """One insight's evidence subset reduced to paper and edge ids, for
    result.json's per-expert `trajectories` list."""
    return {
        "nodes": [{k: p[k] for k in ("label", "paper_id", "title", "year")} for p in ins["papers"]],
        "edges": [{k: e[k] for k in ("edge_id", "from_label", "to_label")} for e in ins["edges"]],
    }


def result_section(progress: dict) -> dict:
    """The Steps 1-3 part of result.json."""
    graph = PaperGraph.from_dict(progress["trajectory"]["graph"])
    insights: List[dict] = progress["insight"]["insights"]
    return {
        "expertise_source": SOURCE_NAME,
        "seed_paper_id": graph.seed_paper_id,
        "paper_graph": graph.result_projection(),
        "retrieved_papers": [n.paper_id for n in graph.nodes],
        # Per-expert evidence subsets: what each expert's expertise was built from.
        "trajectories": [_evidence_projection(ins) for ins in insights],
        "insights": [
            {k: v for k, v in ins.items() if k not in ("papers", "edges")}
            for ins in insights
        ],
        "insight_evidence": [
            {"insight_name": ins.get("insight_name", ""),
             "papers": ins["papers"], "edges": ins["edges"]}
            for ins in insights
        ],
    }

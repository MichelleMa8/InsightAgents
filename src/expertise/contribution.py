"""
Offline expertise backend (EXPERTISE_SOURCE = "contribution"): picks a seed
paper from the local corpus by embedding retrieval, grows a citation-neighbour
PaperGraph around it, and distils the insights with the Semantic Scholar path's
whole-graph insight call.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import config as cfg
import cost_report
from expertise import contribution_graph as cg
from expertise import semantic_scholar as s2
from expertise.semantic_scholar import insight as insight_mod, seed as seed_mod
from expertise.semantic_scholar.paper_graph import PaperGraph, render_nodes
from llm_client import LLMClient
from prompts.contribution_graph_prompts import CG_GRAPH_PROVENANCE
from retrieval.query_expansion import expand_topic_queries
from retrieval.searcher import embed_topic

logger = logging.getLogger(__name__)

SOURCE_NAME = "contribution"

_ROUND_HEADER = "ROUND {level} (citation neighbours of round {prev})"


def run_pretrain(
    spec,
    insight_client: LLMClient,
    progress: dict,
    save: Callable[[], None],
    record_error: Callable[[dict, Optional[str], str], None],
) -> None:
    """Steps 1-3 with per-step checkpointing into `progress`."""
    if progress["insight"] and progress["insight"].get("done"):
        return
    if spec.paper_id:
        raise ValueError("the contribution source starts from a topic, not a paper id")
    topic = spec.topic
    query_vec = None

    # ── 1. Seed paper ─────────────────────────────────────────────────────────
    if not progress["retrieval"]:
        logger.info("[1/3] Selecting seed paper from the local corpus …")
        call_snap = insight_client.call_log_snapshot()
        t0 = time.time()
        query_vec = embed_topic(topic)
        search_queries = expand_topic_queries(topic, insight_client)
        hits = cg.search_seed_candidates(topic, query_vec, search_queries)
        candidates = seed_mod.choose_seed(
            topic, hits, insight_client, stage="cg_seed_select",
        )[:cfg.CG_SEED_MAX_ATTEMPTS]
        calls = insight_client.calls_since(call_snap)

        if not candidates:
            logger.error("No seed paper found for '%s'. Skipping.", topic)
            record_error(progress, "retrieval", "no seed paper found")
            save()
            return

        progress["retrieval"] = {
            "topic": topic,
            "search_queries": search_queries,
            "seed_candidates": candidates,
            "next_candidate_idx": 0,
            "seed_paper_id": None,
        }
        progress["pretrain_stages"].append(cost_report.make_stage_entry(
            stage_id="retrieval",
            label="Step 1 — Seed paper (query expansion + embedding retrieval + LLM pick)",
            calls=calls, duration_sec=time.time() - t0,
            model=insight_client.model if calls else None,
        ))
        save()

    candidates = progress["retrieval"]["seed_candidates"]

    # ── 2. Paper graph ────────────────────────────────────────────────────────
    graph_state = progress["trajectory"]
    if graph_state is None:
        graph_state = progress["trajectory"] = {
            "done": False, "graph": None, "calls": [], "duration_sec": 0.0,
        }
    if not graph_state["done"]:
        logger.info("[2/3] Building citation graph (depth %d, fan-out %s) …",
                    len(cfg.CG_KEEP_PER_LEVEL), cfg.CG_KEEP_PER_LEVEL)
        if query_vec is None:
            query_vec = embed_topic(topic)  # local + deterministic — cheaper to recompute than store
        while progress["retrieval"]["next_candidate_idx"] < len(candidates):
            idx = progress["retrieval"]["next_candidate_idx"]
            seed_paper = candidates[idx]
            call_snap = insight_client.call_log_snapshot()
            t0 = time.time()
            graph = cg.build_paper_graph(topic, seed_paper, query_vec, insight_client)
            graph_state["calls"].extend(insight_client.calls_since(call_snap))
            graph_state["duration_sec"] += time.time() - t0
            ok, reason = cg.validate_graph(graph)
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
            label="Step 2 — Paper graph (recursive citation-neighbour expansion, LLM-filtered)",
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
        insights = insight_mod.generate_insights(
            graph, topic, insight_client,
            provenance=CG_GRAPH_PROVENANCE,
            nodes_text=render_nodes(graph, cfg.S2_INSIGHT_ABSTRACT_CHARS, round_header=_ROUND_HEADER),
            stage="cg_insight",
        )
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


def result_section(progress: dict) -> dict:
    """The Steps 1-3 part of result.json — the S2 layout, plus the search queries
    the seed was retrieved with."""
    return {
        **s2.result_section(progress),
        "expertise_source": SOURCE_NAME,
        "search_queries": progress["retrieval"].get("search_queries", [progress["topic"]]),
    }

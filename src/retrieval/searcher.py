"""
Cosine search over the precomputed contribution-embedding shards. Per-query hits
are rank-fused, deduplicated to papers (each scored by its best contribution),
impact-dampened, and filtered so no two returned papers cite each other.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from config import (
    DATA_DIR,
    RETRIEVAL_CANDIDATE_N,
    RETRIEVAL_FUSION,
    RETRIEVAL_IMPACT_DAMPEN_FLOOR,
    RETRIEVAL_RRF_K,
    RETRIEVAL_TOP_K,
    TRAJ_HUB_IMPACT_CEILING,
)
from graph.impact import get_paper_impact
from graph.loader import load_paper, papers_have_citation_relation
from retrieval.embedder import embed_queries

logger = logging.getLogger(__name__)

EMB_DIR = DATA_DIR / "embeddings"


class RetrievedPaper(NamedTuple):
    """One retrieval hit: the paper, and the contribution of it that scored highest
    against the search queries (why the paper was retrieved)."""
    corpus_id: str
    contribution_id: str
    score: float

_reverse_lut: Optional[Dict[str, Tuple[int, int]]] = None   # contribution_id -> (shard_id, row_index)
_normed_shard_cache: Dict[int, np.ndarray] = {}


def _load_shard(shard_id: int) -> Tuple[np.ndarray, dict]:
    """Load one embedding shard. Returns (matrix float32, lut dict)."""
    npy_path = EMB_DIR / f"shard_{shard_id:06d}.npy"
    lut_path = EMB_DIR / f"shard_{shard_id:06d}.lut.json"
    mat = np.load(npy_path, mmap_mode="r").astype(np.float32)
    with open(lut_path, encoding="utf-8") as f:
        lut = json.load(f)   # contribution_id -> row index
    return mat, lut


def _num_shards() -> int:
    return len(list(EMB_DIR.glob("shard_*.npy")))


def embed_topic(topic: str) -> np.ndarray:
    """Embed a single topic string. Returns an L2-normalised float32 vector (1024,)."""
    return embed_queries([topic])[0]


def _build_reverse_lut() -> Dict[str, Tuple[int, int]]:
    global _reverse_lut
    if _reverse_lut is not None:
        return _reverse_lut
    reverse: Dict[str, Tuple[int, int]] = {}
    for sid in range(_num_shards()):
        lut_path = EMB_DIR / f"shard_{sid:06d}.lut.json"
        with open(lut_path, encoding="utf-8") as f:
            lut = json.load(f)
        for cid, row in lut.items():
            reverse[cid] = (sid, row)
    _reverse_lut = reverse
    logger.info("Built contribution_id reverse lookup over %d entries.", len(reverse))
    return reverse


def get_contribution_embedding(contribution_id: str) -> Optional[np.ndarray]:
    """
    Return the precomputed, L2-normalised embedding for a contribution_id, reusing
    the Step-1 retrieval shards (no fresh embedding call). None if not found.
    """
    loc = _build_reverse_lut().get(contribution_id)
    if loc is None:
        return None
    shard_id, row = loc
    if shard_id not in _normed_shard_cache:
        mat, _ = _load_shard(shard_id)
        norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
        _normed_shard_cache[shard_id] = mat / norms
    return _normed_shard_cache[shard_id][row]


def cosine_search_multi(
    query_mat: np.ndarray,   # shape (n_queries, 1024) float32, rows L2-normalised
    top_n: int = RETRIEVAL_CANDIDATE_N,
) -> List[List[Tuple[str, float]]]:
    """
    Search all shards for every query at once and return one ranked list of
    top_n (contribution_id, score) pairs per query, in the row order of
    query_mat. Query rows must already be L2-normalised (dot product = cosine).

    All queries share a single pass over the shards — the cost of expanding one
    topic into several queries is one extra matrix column per query, not another
    full scan of the corpus (loading + normalising the shards dominates).
    """
    n_shards = _num_shards()
    n_queries = query_mat.shape[0]
    per_query: List[List[Tuple[float, str]]] = [[] for _ in range(n_queries)]

    for sid in range(n_shards):
        mat, lut = _load_shard(sid)
        # Normalise shard rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
        mat_norm = mat / norms
        scores = mat_norm @ query_mat.T   # (N, n_queries)

        # Reverse LUT: row_index -> contribution_id
        idx_to_cid = {v: k for k, v in lut.items()}

        # Take top-n from this shard, per query
        keep = min(top_n, scores.shape[0])
        for qi in range(n_queries):
            col = scores[:, qi]
            top_idx = np.argpartition(col, -keep)[-keep:]
            for idx in top_idx:
                cid = idx_to_cid.get(int(idx))
                if cid:
                    per_query[qi].append((float(col[idx]), cid))

        logger.debug("Shard %d/%d searched (%d queries).", sid + 1, n_shards, n_queries)

    # Global top-n per query
    ranked: List[List[Tuple[str, float]]] = []
    for results in per_query:
        results.sort(reverse=True)
        ranked.append([(cid, score) for score, cid in results[:top_n]])
    return ranked


def _fuse_rankings(
    per_query: List[List[Tuple[str, float]]],
    mode: str = RETRIEVAL_FUSION,
    rrf_k: int = RETRIEVAL_RRF_K,
) -> Tuple[Dict[str, float], Dict[str, Dict[int, int]]]:
    """
    Merge per-query ranked lists into one contribution_id -> fused score map (see
    RETRIEVAL_FUSION in config.py for the two modes and why "rrf" is the default).

    Second return value is provenance for diagnostics: contribution_id ->
    {query index: best rank that query gave it}, so a log line can show which
    query actually surfaced each retrieved paper.
    """
    fused: Dict[str, float] = {}
    hits: Dict[str, Dict[int, int]] = {}

    for qi, results in enumerate(per_query):
        for rank, (cid, score) in enumerate(results, start=1):
            if mode == "rrf":
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            elif mode == "max":
                if cid not in fused or score > fused[cid]:
                    fused[cid] = score
            else:
                raise ValueError(f"Unknown RETRIEVAL_FUSION {mode!r} (expected 'rrf' or 'max')")
            cid_hits = hits.setdefault(cid, {})
            if qi not in cid_hits or rank < cid_hits[qi]:
                cid_hits[qi] = rank

    return fused, hits


def _embed_search_queries(
    queries: Sequence[str],
    topic: str,
    query_vec: Optional[np.ndarray],
) -> np.ndarray:
    """
    Embed every search query into one (n_queries, 1024) matrix, reusing a
    precomputed `query_vec` for the query that is the topic string itself (Step 2
    needs that same vector, so it is embedded once per topic and passed around).
    Everything else is embedded in a single batched call.
    """
    vecs: List[Optional[np.ndarray]] = []
    pending_idx: List[int] = []
    pending: List[str] = []
    for i, query in enumerate(queries):
        if query == topic and query_vec is not None:
            vecs.append(query_vec)
        else:
            vecs.append(None)
            pending_idx.append(i)
            pending.append(query)
    if pending:
        logger.info("Embedding %d search quer%s …", len(pending), "y" if len(pending) == 1 else "ies")
        embedded = embed_queries(pending)
        for slot, row in zip(pending_idx, embedded):
            vecs[slot] = row
    return np.stack(vecs)


def _dampening_factor(impact: int) -> float:
    """
    Dampen only the high-citation tail: papers at or below RETRIEVAL_IMPACT_DAMPEN_FLOOR
    citations get no penalty at all (that floor covers the vast majority of the corpus,
    which is not hub-like), so a paper isn't ranked up merely for having fewer citations
    than another equally-normal paper. Above the floor, dampen continuously by how far
    past it the citation count reaches, capping the citation count at
    TRAJ_HUB_IMPACT_CEILING first — so a true mega-hub (tens of thousands+ citations)
    is dampened no harder than a paper right at the ceiling, and can still surface if
    its similarity is high enough, instead of being excluded outright regardless of
    topical relevance.
    """
    capped_impact = min(impact, TRAJ_HUB_IMPACT_CEILING)
    excess = max(0, capped_impact - RETRIEVAL_IMPACT_DAMPEN_FLOOR)
    return 1.0 + math.log1p(excess)


def retrieve_papers(
    topic: str,
    top_k: int = RETRIEVAL_TOP_K,
    candidate_n: int = RETRIEVAL_CANDIDATE_N,
    query_vec: Optional[np.ndarray] = None,
    queries: Optional[Sequence[str]] = None,
) -> List[RetrievedPaper]:
    """
    Retrieve the top_k papers relevant to a topic string, each paired with the
    contribution of it that scored highest.

    Steps:
      1. Embed every search query with Qwen3 (reusing a precomputed query_vec for
         the topic string itself)
      2. Cosine search → top candidate_n contributions per query
      3. Fuse the per-query rankings into one score per contribution
      4. Deduplicate to paper level (keep best-scoring contribution per paper)
      5. Filter pairs that have citation relationships
      6. Return top_k (corpus_id, contribution_id, score) triples

    `queries`: the search queries to run, normally `expand_topic_queries(topic, …)`
    output (the topic string first, then its single-meaning expansions). None or
    empty falls back to searching the topic string alone.
    """
    if query_vec is None:
        logger.info("Embedding topic: %s", topic)
        query_vec = embed_topic(topic)

    search_queries = list(queries) if queries else [topic]
    query_mat = _embed_search_queries(search_queries, topic, query_vec)

    logger.info("Searching %d shards with %d quer%s …",
                _num_shards(), len(search_queries), "y" if len(search_queries) == 1 else "ies")
    per_query = cosine_search_multi(query_mat, top_n=candidate_n)
    fused, hits = _fuse_rankings(per_query)
    logger.info("Fused %d unique contribution(s) from %d x %d results (fusion=%s).",
                len(fused), len(search_queries), candidate_n, RETRIEVAL_FUSION)

    # Deduplicate: keep best-scoring contribution per paper, and track its cid
    paper_best: dict[str, tuple[float, str]] = {}  # corpus_id -> (score, best_cid)
    for cid, score in fused.items():
        corpus_id = cid.split(".")[0]
        if corpus_id not in paper_best or score > paper_best[corpus_id][0]:
            paper_best[corpus_id] = (score, cid)

    # Rank papers; continuously dampen high-citation general papers instead of a step
    # penalty, since impact ranges from ~10 to 60k+ and a flat ×0.5 can't offset that spread.
    # Dampening only kicks in above RETRIEVAL_IMPACT_DAMPEN_FLOOR and saturates at
    # TRAJ_HUB_IMPACT_CEILING (see _dampening_factor) — most of the corpus sits at or near
    # zero citations, so an undampened log1p(impact) would systematically favor low/zero-
    # citation papers over equally-normal ones; the saturation cap means even a mega-hub
    # (e.g. BERT/Transformer-tier) isn't excluded outright, but has to overcome the max
    # dampening penalty on similarity alone to rank highly.
    ranked_papers = sorted(
        paper_best.items(),
        key=lambda x: x[1][0] / _dampening_factor(get_paper_impact(x[0])),
        reverse=True,
    )

    # Mutual-citation filtering: greedily select top_k papers with no mutual citations
    selected: List[RetrievedPaper] = []
    selected_papers: List[dict] = []

    for corpus_id, (fused_score, best_cid) in ranked_papers:
        try:
            paper = load_paper(corpus_id)
        except FileNotFoundError:
            continue

        # Check against all already-selected papers
        conflict = any(papers_have_citation_relation(paper, sp) for sp in selected_papers)
        if conflict:
            logger.debug("Skipping %s due to mutual citation with selected papers.", corpus_id)
            continue

        selected.append(RetrievedPaper(corpus_id, best_cid, fused_score))
        selected_papers.append(paper)
        # Which query surfaced this paper, and at what rank — the diagnostic that
        # says whether expanding the topic actually widened the retrieved set.
        logger.info(
            "  #%d %s seed=%s (score=%.4f) via %s",
            len(selected), corpus_id, best_cid, fused_score,
            ", ".join(
                f"{search_queries[qi]!r}@{rank}"
                for qi, rank in sorted(hits.get(best_cid, {}).items())
            ) or "?",
        )

        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        logger.warning(
            "Only %d/%d papers retrieved for topic '%s' after citation filtering.",
            len(selected), top_k, topic,
        )

    return selected

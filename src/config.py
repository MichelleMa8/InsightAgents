"""InsightAgents configuration.

Sections are ordered by how often they are changed: the first few are the
usual knobs, everything below "Advanced" can normally stay at its default.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")


# ══════════════════════════════════════════════════════════════════════════════
# Main settings
# ══════════════════════════════════════════════════════════════════════════════

# ── Models ────────────────────────────────────────────────────────────────────
LLM_PROVIDER  = "openai"         # "openai" | "anthropic"; must match the models below
DEBATE_MODEL  = "gpt-5.6-luna"   # all debate agents (experts, moderator, critic)
INSIGHT_MODEL = "gpt-5.6-luna"   # seed selection, graph filtering, insight generation

EVAL_PROVIDER = "anthropic"        # pairwise judge; CLI --provider overrides
EVAL_MODEL    = "claude-opus-4-8"  # CLI --model overrides

# ── Expertise source ──────────────────────────────────────────────────────────
# "semantic_scholar" (online): live S2 API, recommendation graph.
# "contribution" (offline): local Scientific Contribution Graph, citation graph.
# CLI --expertise-source online|offline overrides.
EXPERTISE_SOURCE = "semantic_scholar"

NUM_AGENTS = 3   # expert agents = insights distilled from the paper graph

# Papers kept per parent per expansion round; length = graph depth.
# [5, 3, 2] -> at most 1 + 5 + 15 + 30 = 51 papers.
S2_KEEP_PER_LEVEL = [5, 3, 2]   # online
CG_KEEP_PER_LEVEL = [5, 3, 2]   # offline

# ── Debate ────────────────────────────────────────────────────────────────────
# Stage 1 problem determination -> Stage 2 debate -> Stage 3 final rewrite.
PROBLEM_TURNS      = 2       # Stage 1 turns: 1 = statement only, 2 = + blind endorsement vote
DEBATE_TURNS       = 3       # Stage 2 turns, minimum 3 (propose / clash / converge)
CRITIC_ENABLED     = True    # external critic after each Stage 2 turn except the last
EARLY_STOP_ENABLED = True    # end Stage 2 early once the moderator reports convergence
ORDER_MODE         = "rule"  # clash-turn order: "rule" (attacked experts first) | "llm"

# Opening turns written without seeing earlier speakers (env BLIND_OPENING_TURNS=1).
BLIND_OPENING_TURNS = os.getenv("BLIND_OPENING_TURNS", "0") == "1"

MAX_TOKENS_PER_SPEECH     = 2500   # one expert's speech within a turn
FINAL_PROPOSAL_MAX_TOKENS = 2000   # Stage 3 write-up

# ── Pairwise evaluation ───────────────────────────────────────────────────────
EVAL_ROUNDS      = 1       # rounds per pair; each round judges both orders (A,B) and (B,A)
EVAL_MAX_TOKENS  = 10000   # shared with the judge's thinking; much less truncates answers
EVAL_MAX_RETRIES = 2       # re-asks when a judgment fails to parse

# ── Paths and API keys ────────────────────────────────────────────────────────
DATA_DIR      = ROOT / "data" / "sci-cont-graph" / "data"   # SCG release (offline mode)
OUTPUT_DIR    = ROOT / "outputs"
S2_CACHE_DIR  = ROOT / "data" / "s2_cache"                  # Semantic Scholar response cache
ACL_CACHE_DIR = ROOT / "data" / "acl_cache"                 # ACL abstract index (offline mode)

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
S2_API_KEY         = os.getenv("S2_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")   # empty = SDK default endpoint
OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "")      # e.g. Azure / vLLM / proxy


# ══════════════════════════════════════════════════════════════════════════════
# Advanced settings (usually left at their defaults)
# ══════════════════════════════════════════════════════════════════════════════

# ── LLM client ────────────────────────────────────────────────────────────────
# "responses" is required for GPT-5.x reasoning models with tool calls (critic,
# endorsement vote); use "chat" only for OpenAI-compatible servers without
# /v1/responses.
OPENAI_API_STYLE = "responses"   # "responses" | "chat"

# "none" | "low" | "medium" | "high" | "xhigh" | "max", or "" for the model default.
OPENAI_REASONING_EFFORT = "medium"

# Extra output tokens added on reasoning models on top of every answer budget,
# since the hidden reasoning chain shares max_tokens. 0 = no reserve.
REASONING_TOKEN_RESERVE = 2000

MAX_TOKENS_PER_CRITIC = 4000
CRITIC_MAX_RETRIES    = 1

# Re-asks for malformed free-form JSON (insight generation, topic derivation from a seed paper).
INSIGHT_JSON_MAX_RETRIES = 1

# ── Online: Semantic Scholar API ──────────────────────────────────────────────
S2_API_BASE             = "https://api.semanticscholar.org"
S2_TIMEOUT              = 30.0   # seconds per request
S2_MIN_REQUEST_INTERVAL = 1.5    # process-wide pacing; a key allows 1 request/s
S2_MAX_RETRIES          = 8      # on 429 / 5xx, exponential backoff:
S2_BACKOFF_BASE         = 1.0    #   min(MAX, BASE * 2**attempt) + jitter
S2_BACKOFF_MAX          = 30.0
S2_CACHE_ENABLED        = True   # delete S2_CACHE_DIR to force fresh data

# ── Online: seed paper ────────────────────────────────────────────────────────
S2_SEARCH_LIMIT            = 10                 # paper-search results per date window
S2_SEED_MAX_AGE_DAYS_STEPS = [365, 730, None]   # recency windows tried in order (None = any date)
S2_REFERENCE_DATE          = "2026-09-23"       # "today" for the windows; None = real today
S2_SEED_SHORTLIST          = 5                  # top results the LLM picks the seed from (1 = take rank 1)
S2_SEED_SELECT_MAX_TOKENS  = 400
S2_SEED_SELECT_MAX_RETRIES = 2
S2_SEED_MAX_ATTEMPTS       = 3                  # seeds tried when a graph fails validation
S2_TOPIC_FROM_PAPER_MAX_TOKENS = 200            # topic derived from a seed paper without one

# ── Online: graph expansion ───────────────────────────────────────────────────
S2_RECS_PER_PAPER        = 20         # /recommendations results per paper
S2_REC_POOL              = "recent"   # "recent" | "all-cs"
S2_FILTER_MAX_TOKENS     = 800        # per-parent LLM filter call
S2_FILTER_MAX_RETRIES    = 2
S2_FILTER_ABSTRACT_CHARS = 1800       # abstract truncation in the filter prompt
S2_MIN_GRAPH_NODES       = 8          # smaller graphs fail validation

# ── Insight generation (shared by both modes) ─────────────────────────────────
# One call over the whole graph returns NUM_AGENTS insights, each with its own
# evidence subset; an insight below the floors is re-asked.
S2_NUM_INSIGHTS                = NUM_AGENTS
S2_INSIGHT_ABSTRACT_CHARS      = 1800
S2_INSIGHT_MAX_TOKENS          = 8000
S2_INSIGHT_MIN_CLAIMS          = 3   # lower it for weak generator models
S2_INSIGHT_MIN_EVIDENCE_PAPERS = 2
S2_INSIGHT_MIN_EVIDENCE_EDGES  = 0

# ── Offline: corpus scope ─────────────────────────────────────────────────────
# SCG 1.0 effectively ends in 2025. A graph whose newest paper is older than
# CORPUS_MAX_YEAR - SUBGRAPH_MAX_STALENESS_YEARS is rejected (None = no check).
CORPUS_MAX_YEAR = 2025
SUBGRAPH_MAX_STALENESS_YEARS = 1
SUBGRAPH_MIN_RECENT_YEAR = (
    None if SUBGRAPH_MAX_STALENESS_YEARS is None
    else CORPUS_MAX_YEAR - SUBGRAPH_MAX_STALENESS_YEARS
)

# ── Offline: seed retrieval ───────────────────────────────────────────────────
RETRIEVAL_TOP_K               = NUM_AGENTS
RETRIEVAL_CANDIDATE_N         = 100     # cosine hits kept per search query
RETRIEVAL_IMPACT_DAMPEN_FLOOR = 25      # citation counts above this are dampened
TRAJ_HUB_IMPACT_CEILING       = 20000   # hub cutoff: caps retrieval dampening; citation neighbours above it are skipped
RETRIEVAL_FUSION              = "rrf"   # merge per-query rankings: "rrf" | "max"
RETRIEVAL_RRF_K               = 10      # RRF rank offset; larger = favour agreement across queries

# LLM rewrites the topic into extra single-meaning search queries (the topic
# itself is always searched too).
QUERY_EXPANSION_ENABLED     = False
NUM_SEARCH_QUERIES          = 5
QUERY_EXPANSION_MAX_RETRIES = 1
QUERY_EXPANSION_MAX_TOKENS  = 600

CG_SEED_SEARCH_LIMIT   = 10   # retrieved candidates; the LLM picks from the first S2_SEED_SHORTLIST
CG_SEED_MAX_ATTEMPTS   = 3    # seeds tried when a graph fails validation
CG_SEED_MIN_YEAR_STEPS = [CORPUS_MAX_YEAR - 1, CORPUS_MAX_YEAR - 3, None]   # newest group first

# ── Offline: graph expansion ──────────────────────────────────────────────────
CG_CANDIDATES_PER_PAPER = 20     # citation neighbours shown to the per-parent filter
CG_SIMILARITY_THRESHOLD = 0.30   # min topic cosine to keep a neighbour
CG_FILTER_MAX_TOKENS    = 800
CG_FILTER_MAX_RETRIES   = 2
CG_ABSTRACT_CHARS       = 1800
CG_MIN_GRAPH_NODES      = 8

# ── API prices (USD per million tokens, cost estimates only) ──────────────────
# Buckets are mutually exclusive: `input` counts uncached prompt tokens only.
# Lookup: exact model id, then longest matching prefix; unknown models log
# token counts without a cost.
API_PRICES_PER_MTOK = {
    # Anthropic
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},

    # OpenAI: GPT-5.6
    "gpt-5.6-sol":   {"input":  5.00, "output":  30.00, "cache_write":  5.00, "cache_read": 0.50},
    "gpt-5.6-terra": {"input":  2.00, "output":  12.00, "cache_write":  2.00, "cache_read": 0.20},
    "gpt-5.6-luna":  {"input":  0.20, "output":   1.20, "cache_write":  0.20, "cache_read": 0.02},

    # OpenAI: GPT-5.5 / 5.4 / 5.2 / 5.1 (pro models: no caching)
    "gpt-5.5":       {"input":  5.00, "output":  30.00, "cache_write":  5.00, "cache_read": 0.50},
    "gpt-5.5-pro":   {"input": 30.00, "output": 180.00, "cache_write": 30.00, "cache_read": 30.00},
    "gpt-5.4":       {"input":  2.50, "output":  15.00, "cache_write":  2.50, "cache_read": 0.25},
    "gpt-5.4-mini":  {"input":  0.75, "output":   4.50, "cache_write":  0.75, "cache_read": 0.075},
    "gpt-5.4-nano":  {"input":  0.20, "output":   1.25, "cache_write":  0.20, "cache_read": 0.02},
    "gpt-5.4-pro":   {"input": 30.00, "output": 180.00, "cache_write": 30.00, "cache_read": 30.00},
    "gpt-5.2":       {"input":  1.75, "output":  14.00, "cache_write":  1.75, "cache_read": 0.175},
    "gpt-5.1":       {"input":  1.25, "output":  10.00, "cache_write":  1.25, "cache_read": 0.125},

    # OpenAI: GPT-5
    "gpt-5":         {"input":  1.25, "output":  10.00, "cache_write":  1.25, "cache_read": 0.125},
    "gpt-5-mini":    {"input":  0.25, "output":   2.00, "cache_write":  0.25, "cache_read": 0.025},
    "gpt-5-nano":    {"input":  0.05, "output":   0.40, "cache_write":  0.05, "cache_read": 0.005},

    # OpenAI: GPT-4.1 / 4o
    "gpt-4.1":       {"input":  2.00, "output":   8.00, "cache_write":  2.00, "cache_read": 0.50},
    "gpt-4.1-mini":  {"input":  0.40, "output":   1.60, "cache_write":  0.40, "cache_read": 0.10},
    "gpt-4.1-nano":  {"input":  0.10, "output":   0.40, "cache_write":  0.10, "cache_read": 0.025},
    "gpt-4o":        {"input":  2.50, "output":  10.00, "cache_write":  2.50, "cache_read": 1.25},
    "gpt-4o-mini":   {"input":  0.15, "output":   0.60, "cache_write":  0.15, "cache_read": 0.075},

    # OpenAI: o-series
    "o1":            {"input": 15.00, "output":  60.00, "cache_write": 15.00, "cache_read": 7.50},
    "o3":            {"input":  2.00, "output":   8.00, "cache_write":  2.00, "cache_read": 0.50},
    "o3-mini":       {"input":  1.10, "output":   4.40, "cache_write":  1.10, "cache_read": 0.55},
    "o4-mini":       {"input":  1.10, "output":   4.40, "cache_write":  1.10, "cache_read": 0.275},
}

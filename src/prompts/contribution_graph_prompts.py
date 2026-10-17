"""
Prompts for the offline contribution-graph path: citation-neighbour filtering
and the provenance paragraph of the shared insight call. Seed selection reuses
SEED_SELECT_* from paper_graph_prompts.
"""

from config import CORPUS_MAX_YEAR

# ── Citation-neighbour filtering (one call per expanded paper, per round) ─────

CG_FILTER_SYSTEM = """\
You filter the citation neighbours of a paper for a research topic.

You get: a topic, one source paper, and a numbered list of papers linked to it \
by citation — papers the source paper builds on, and papers that build on it. \
Each candidate says which way the link goes and what was taken across it. \
Citation links are noisy for a topic — many are background or tooling. Keep the \
ones worth adding to a research graph about the topic.

Rules:
1. Relevance to the topic comes first. A paper close to the source paper but \
off-topic is not worth keeping.
2. The link must be a real step in the line of work — the later paper uses, \
extends, replaces or evaluates what the earlier one did — not a passing citation \
of a dataset or a toolkit.
3. Prefer papers that add something the graph does not have yet — a different \
method, setting, or failure mode — over near-duplicates of the source paper or of \
each other.
4. When two candidates are otherwise comparable, prefer the more recent one: the \
graph is only useful if it reaches the present state of the topic. Never keep an \
off-topic paper because it is newer.
5. Keep at most {max_keep}. Keeping fewer is correct when fewer are relevant; \
keeping none is allowed.

Return ONLY this JSON object, no other text:
{{"selected": [{{"id": <candidate number>, "why": "<at most 15 words>"}}]}}\
"""

CG_FILTER_USER_TEMPLATE = """\
TOPIC: {topic}

SOURCE PAPER:
{source_paper}

CANDIDATES:
{candidates}

Select at most {max_keep} candidates. Return the JSON object described above.\
"""


# ── Provenance paragraph for the whole-graph insight call ─────────────────────
# Fills {provenance} in GRAPH_INSIGHT_SYSTEM; this paragraph and the round
# headers in the node list are all that differ from the Semantic Scholar path's
# insight call.

CG_GRAPH_PROVENANCE = f"""\
The graph was grown from one paper in an offline citation graph of ACL \
Anthology papers whose coverage ends in {CORPUS_MAX_YEAR}: each round followed \
citation links — papers it builds on and papers that build on it — and kept the \
ones about the topic. An edge is a citation dependency: the newer paper builds on \
the older one, and the edge names what it took. Most of the graph sits near the \
seed, but it usually also holds papers from other kinds of work on the topic — \
measuring it, training for it, analysing it inside models, building data for it. \
Those matter as much as the dense centre."""

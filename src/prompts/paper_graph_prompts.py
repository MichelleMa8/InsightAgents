"""
Prompts for the Semantic Scholar expertise path (config.EXPERTISE_SOURCE =
"semantic_scholar"): seed paper selection, recommendation filtering, topic
derivation from a seed paper, and the single whole-graph insight call. Seed
selection and the insight call are shared with the contribution-graph path.
"""

# ── Seed paper selection (one call per topic) ─────────────────────────────────

SEED_SELECT_SYSTEM = """\
You pick the starting paper for a research graph about a topic.

You get a topic and a numbered list of recent papers that a search returned for \
it. Search ranks by word overlap, so the top hit is often a paper from a \
different research area that happens to share the topic's words. Your job is to \
pick the one paper that is really about the topic.

Rules:
1. The paper must be about the topic itself, not merely share words with it. If a \
term in the topic means something different in that paper's field, it is the \
wrong pick.
2. Prefer a paper whose main problem IS the topic over one that only touches it \
as a component.
3. Prefer a paper that sits in the middle of the topic's literature over a narrow \
or unusual application of it — the whole graph is grown from this paper.
4. Citations, venue and date only break ties.
5. Pick exactly one, even if none is a perfect fit; say so in `why` when the fit \
is poor.

Return ONLY this JSON object, no other text:
{"pick": <candidate number>, "why": "<at most 20 words>"}\
"""

SEED_SELECT_USER_TEMPLATE = """\
TOPIC: {topic}

CANDIDATES:
{candidates}

Pick the one paper this topic's research graph should be grown from. Return the \
JSON object described above.\
"""


# ── Recommendation filtering (one call per expanded paper, per round) ─────────

REC_FILTER_SYSTEM = """\
You filter paper recommendations for a research topic.

You get: a topic, one source paper, and a numbered list of papers that Semantic \
Scholar recommends for that source paper. Recommendations are noisy — many are \
only loosely related. Keep the ones worth adding to a research graph about the \
topic.

Rules:
1. Relevance to the topic comes first. A paper that is close to the source paper \
but off-topic is not worth keeping.
2. The paper must also be a real neighbour of the source paper's line of work: it \
addresses the same problem, uses or replaces the same kind of method, or evaluates \
the same thing.
3. Prefer papers that add something the graph does not have yet — a different \
method, setting, or failure mode — over near-duplicates of the source paper or of \
each other.
4. Keep at most {max_keep}. Keeping fewer is correct when fewer are relevant; \
keeping none is allowed.

Return ONLY this JSON object, no other text:
{{"selected": [{{"id": <candidate number>, "why": "<at most 15 words>"}}]}}\
"""

REC_FILTER_USER_TEMPLATE = """\
TOPIC: {topic}

SOURCE PAPER:
{source_paper}

CANDIDATES:
{candidates}

Select at most {max_keep} candidates. Return the JSON object described above.\
"""


# ── Topic string from a seed paper (seed-paper input without a topic) ────────

TOPIC_FROM_PAPER_SYSTEM = """\
You name the research topic of a paper.

Write one short noun phrase (6-14 words) that names the research problem the \
paper works on — not the paper's method, not its title reworded. It should read \
like a topic another researcher could work on independently.

Return ONLY this JSON object, no other text:
{"topic": "<the noun phrase>"}\
"""

TOPIC_FROM_PAPER_USER_TEMPLATE = """\
PAPER:
{paper}

Name this paper's research topic.\
"""


# ── Whole-graph insight generation ───────────────────────────────────────────
#
# An insight is background knowledge, not a methodological move: what a
# researcher who has lived inside one line of work knows about where it stands.
# An expert handed a move tends to propose that move; the example below carries
# the distinction.

INSIGHT_DEFINITION = """\
Each insight is one line of work seen from the inside: what problem it keeps \
running into, what has been tried against it and how that has shifted, what it \
takes for granted, and where it is stuck. Its purpose is to inform someone who \
will then think freely about the topic — it should open up what they can see, \
not hand them a move to reuse.

Alongside the reading, keep the payload: the specific findings the stance rests \
on, with the method names, benchmarks and numbers as reported. The reading is \
what an insider believes; the findings are why they believe it — and they are \
the one part of expertise a newcomer cannot invent.

Two versions of an insight about the same three papers on retrieval-augmented \
generation:
  Recipe: "Retrievers are repeatedly fine-tuned on downstream task signal to \
close the gap between retrieval relevance and answer usefulness."
  Background: "This line of work keeps finding that what a retriever ranks highly \
is not what the generator needs — the two are trained on different notions of \
relevance. Closing the gap by training the retriever on the generator's signal \
has helped but couples the two tightly, and nobody has yet asked whether the \
generator should instead learn to use imperfect retrieval."
The second tells a reader where the frontier is and leaves them free to decide \
what to do about it."""

# Where the papers came from (the Semantic Scholar recommendation graph). The
# offline path substitutes CG_GRAPH_PROVENANCE (prompts/contribution_graph_prompts.py).
GRAPH_PROVENANCE = """\
The graph was grown from one recent paper: each round asked Semantic Scholar for \
related papers and kept the ones about the topic. Most of it sits near the seed, \
but the graph usually also holds papers from other kinds of work on the topic — \
training it in, steering it inside the model, watching it in agents, building \
data for it. Those matter as much as the dense centre."""

GRAPH_INSIGHT_SYSTEM = f"""\
You read a graph of related papers and write {{num_insights}} pieces of \
background knowledge from it — the understanding a researcher has after living \
inside one line of work for a while.

{{provenance}}

FIRST, SORT THE GRAPH

Before writing anything, sort the papers into the distinct lines of work the \
graph contains. A line of work is a kind of research — measuring something, \
training for it, locating it inside models, studying it in deployed agents, \
building the data — not a different angle on one question. Then choose the \
{{num_insights}} most different lines that have at least two papers each. If \
the graph only carries fewer than {{num_insights}} kinds of work, the remaining \
insight may be a second line inside the largest kind; say so in \
grounding_summary.

WHAT YOU ARE WRITING

{INSIGHT_DEFINITION}

Write each insight for a researcher from a different sub-field: name the line \
the way the field would name it, say what it keeps finding and where it is \
stuck in plain words, and keep the method names, benchmarks and numbers the \
papers report — saying in a few words what each named thing is, so a newcomer \
can use the finding without having read the paper. The papers are your evidence \
for what the line has found; the insight is about the line, not about the papers.

OUTPUT

Return only this JSON, using the [P#] labels exactly as given:

{{{{
  "insights": [
    {{{{
      "insight_name": "2-5 word name for this line of work, as the field would say it",
      "core_insight": "3-4 sentences: what this line keeps running into, what has been tried, what it assumes, where it is stuck (max 90 words)",
      "historical_tension": "the trade-off or pressure that keeps it moving (max 40 words)",
      "claims": [
        {{{{"label": "P3",
          "finding": "one specific result that paper reports and this reading rests on, with the names and numbers it uses (max 35 words)"}}}}
      ],
      "grounding_summary": "which papers this reading rests on, and how much of this line the graph actually shows (max 60 words)",
      "papers": [
        {{{{"label": "P3",
          "neutral_description": "what this paper did, factually (max 25 words)",
          "role_in_insight": "its place in this line of work (max 25 words)"}}}}
      ],
      "edges": [
        {{{{"edge_id": "P3→P7",
          "transition_summary": "what changed between the two papers and why (max 30 words)",
          "role_in_abstraction": "what this step shows about the line of work (max 25 words)"}}}}
      ]
    }}}}
  ]
}}}}

Every insight rests on at least {{min_papers}} papers that are actually in the \
graph; list the edges between them that exist in the EDGES list, and leave \
"edges" empty when there are none — a line of work at the graph's edge often \
has papers but no recorded steps. Use only [P#] labels and edge ids from the \
lists. Give each insight 4-6 claims, each from a different paper where the \
evidence allows it. Don't invent papers, methods or results."""

GRAPH_INSIGHT_USER_TEMPLATE = """\
TOPIC: {topic}

PAPERS IN THE GRAPH:
{nodes_text}

EDGES (older paper → newer paper):
{edges_text}

Write {num_insights} different insights from this graph. Return the JSON object \
described above.\
"""

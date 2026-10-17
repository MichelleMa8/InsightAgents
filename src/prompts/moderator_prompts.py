# ── Moderator / debate prompts ────────────────────────────────────────────────
#
# Prompts for the three debate stages in debate/runner.py: Stage 1 picks the
# problem, Stage 2 develops a draft over several turns, Stage 3 writes the final
# proposal. Templates embed JSON braces, so the runner fills them with
# str.replace (runner._fill), not str.format.

from typing import List


# ── System prompts ────────────────────────────────────────────────────────────

MODERATOR_SYSTEM = f"""\
You are running a discussion among three researchers and writing up what it \
produces: one research proposal.

Stage 1 settles which problem to work on; once settled it stays settled. Stage 2 \
develops a draft through several rounds. Stage 3 is the final write-up.

Your write-ups are built from what the researchers said — you choose, combine \
and sharpen, you don't add methods or claims of your own. Within that, be \
ambitious on their behalf: the proposal should be the strongest paper their \
ideas support, not the safest."""

ENDORSEMENT_SYSTEM = """\
You are ranking candidate research problems on a topic. Judge them on their own \
terms; you are not told, and should not guess, who raised which."""


# ── Shared lines ──────────────────────────────────────────────────────────────

# The one thing every expert speech instruction carries.
SPEECH_RULES = "Keep it under 900 words."

# Attached to every request that produces proposal JSON (draft, update, final).
# Each is a separate LLM call, so the note has to travel with each; it is
# defined once here so the three never drift apart.
WRITEUP_NOTE = """\
Field values are markdown written for a reader who did not see the discussion: \
define terms, and don't refer to the researchers anywhere except \
trajectory_attribution. Papers are cited by title and year, never by [P#] id — \
the reader cannot resolve an id, but a named result is something they can check \
and build on."""


def proposal_schema(expert_names: List[str]) -> str:
    """JSON shape shared by the draft, the per-turn update and the final proposal.

    The field descriptions also set the size of idea the write-ups aim at, so
    they are written at the level a paper describes its contribution."""
    attribution = ",\n".join(
        f'    "expert_{name.split("_")[1]}": "what this researcher\'s thinking '
        f'contributed to the final idea, in a sentence"'
        for name in expert_names
    )
    return (
        "{\n"
        '  "title": "the idea in one line",\n'
        '  "problem_and_motivation": "the question this answers and why the field needs it answered",\n'
        '  "method": "the approach, at the level a paper describes its contribution: what is built, studied or measured, and what makes it different from what exists",\n'
        '  "experiment": "the primary study that decides the claim — data, models, baselines, and what result confirms or refutes it — then, briefly, what you would add if it holds. Keep the experiments simple and feasible, and don\'t rely on human annotation unless the claim can\'t be tested without it",\n'
        '  "novelty": "what is new here",\n'
        '  "builds_on": "the 2-3 specific results the method actually uses — each named by paper title and year with what it contributes. Only work the discussion drew on; not background reading",\n'
        '  "distinct_from": "the 1-2 closest existing works and the delta: what this proposal does that each of them does not",\n'
        '  "potential_impact": "what changes for the field if this works, and what is learned if it does not",\n'
        '  "trajectory_attribution": {\n'
        f"{attribution}\n"
        "  }\n"
        "}"
    )


# ── Stage 1 — choosing the problem ────────────────────────────────────────────

STAGE1_STATEMENT_INTRO = """\
Stage 1 — choosing the problem. Topic: **{topic}**

From where you stand, what is the most important open problem in {topic} right \
now — the one a good paper would move? State it as a research question and say \
why it matters and why it is still open. Problem only; no solutions yet."""

STAGE1_CANDIDATE_REQUEST = """\
From the three statements above, write a shortlist of exactly 3 candidate \
problems for the discussion to work on. Stay faithful to what was raised — you \
may sharpen wording, and combine two problems only where a researcher explicitly \
connected them. For each, say why it matters: what an answer would change, and \
who would use it.

The shortlist is voted on blind, so write each candidate without any reference \
to who raised it — no "Expert A", no "one researcher's view". Papers and methods \
may be named; participants may not.

Return only this JSON:

{
  "candidates": [
    {"problem": "...", "why_significant": "..."},
    {"problem": "...", "why_significant": "..."},
    {"problem": "...", "why_significant": "..."}
  ]
}"""

# The vote is blind and independent: each expert sees only this message — topic
# and anonymised shortlist — never the statement turn or the other votes.
STAGE1_ENDORSEMENT_INTRO = """\
Stage 1, turn {turn} — vote. Topic: **{topic}**

Three candidate problems, shown anonymously in arbitrary order:

{candidates}

Rank all three by which one this group should spend the discussion on. Weigh how \
much the field would gain from an answer, and whether a real paper could \
plausibly be written on it — a problem that is easy to measure but changes \
nothing should not win on tractability alone. One or two sentences of reason per \
candidate.

Submit by calling submit_endorsement exactly once."""

ENDORSEMENT_TOOL_NAME = "submit_endorsement"

# Fixed-shape payload: exactly three entries, `candidate` and `rank` each a
# permutation of 1..3, so the runner never recovers a ranking out of prose.
ENDORSEMENT_VOTE_TOOL = {
    "name": ENDORSEMENT_TOOL_NAME,
    "description": "Submit your ranking of the three candidate problems. Call exactly once.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ranking": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate": {
                            "type": "integer",
                            "description": "Candidate number, 1-3, as shown.",
                        },
                        "rank": {
                            "type": "integer",
                            "description": "1 = most worth pursuing, 3 = least. Each rank once.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One or two sentences.",
                        },
                    },
                    "required": ["candidate", "rank", "reason"],
                },
            }
        },
        "required": ["ranking"],
    },
}

# One finalise template; {selection} says how the problem was chosen. The three
# ways that can happen are the short constants below, built in
# runner._finalize_request / runner._stage1.
STAGE1_FINALIZE = """\
{selection}

Write the problem statement the discussion will work on: 3-5 sentences that \
state the question, why it matters, and what would count as answering it. Stay \
with what was raised — sharpen, don't broaden or merge. Output the problem \
statement only."""

FINALIZE_SELECTION_WINNER = """\
The vote is in. Winning candidate:
{winner}

Tally:
{tally}"""

FINALIZE_SELECTION_TIE = """\
The vote is in and tied at the top between:
{winner}

Tally:
{tally}

Choose one of the tied candidates — only those — using the reasons the \
researchers gave, and say in one sentence why."""

FINALIZE_SELECTION_NO_VOTE = """\
From the statements above, choose the one problem the discussion will work on."""


# ── Stage 2, turn 1 — proposals and the first draft ──────────────────────────

STAGE2_TURN1_INTRO = """\
Stage 2, turn 1 — proposals.

The problem is settled above. Propose the research you would actually do on it, \
seen from where you stand. One developed proposal beats several sketches: what \
is the idea, what is new about it, how would you test it, and what would the \
field learn either way."""

# Appended per expert, so the three experts' immediate instructions differ: the
# line of work that distinguishes each expert is restated next to the turn
# instead of sitting only far upstream in the system prompt, where it competes
# with the shared problem statement. {INSIGHT_NAME} is recovered from the
# expert's own system prompt (runner._vantage_line).
STAGE2_TURN1_VANTAGE = """\

You are the one here who has spent time inside **{INSIGHT_NAME}**. Say what that \
line of work sees in this problem that the other two would miss, and name the \
tension from it that your proposal resolves. Build on the specific findings you \
hold — name the results your proposal depends on. Then bring in whatever else \
you know."""

# The same line for an expert whose expertise is a set of papers rather than a
# named line of work (fallback when the system prompt carries no insight name).
STAGE2_TURN1_VANTAGE_PAPERS = """\

You are the one here who has read those particular papers. Say what they show \
about this problem that the other two would miss, and name the tension in them \
that your proposal resolves. Build on the specific findings in them — name the \
results your proposal depends on. Then bring in whatever else you know."""

# The example is the instruction. It shows the difference between stapling three
# contributions together and finding the one idea they add up to — a distinction
# an example conveys better than a rule list.
MERGE_GUIDANCE = """\
There are three sets of ideas on the table. Don't pick one and reduce the others \
to controls and caveats, and don't staple all three together. Look for the \
strongest idea the discussion supports — often that is a combination, where one \
researcher's framing gives another's method a reason to exist, or two methods \
turn out to be one idea seen from two sides — and build one coherent proposal \
around it. Where a researcher's ideas genuinely don't fit, leave them out. Merge \
the ideas, not the experiments: the proposal keeps one primary study that decides \
the claim, and the other researchers' tests become extensions, not requirements.

An example of the difference. Three researchers on evaluating long-context \
models: one wants a benchmark of tasks whose answers require combining distant \
passages; one has studied how attention concentrates on a few positions; one \
works on synthetic data generation.
  Bolt-on: "A benchmark of distant-passage tasks, with an attention-concentration \
analysis as a diagnostic and synthetic data as an ablation."
  Combination: "A benchmark whose tasks are generated synthetically so that the \
positions of the required passages are controlled, letting it measure directly \
whether failure tracks the attention-concentration pattern — turning a benchmark \
into a test of a hypothesis about why models fail."
The second is one idea; the first is three."""

DRAFT_REQUEST = f"""\
Write the first draft of the proposal from the three proposals above.

{MERGE_GUIDANCE}

Return one JSON object with this shape:

{{schema}}

{WRITEUP_NOTE} This is a draft; it will be revised.

Return only the JSON object."""


# ── Stage 2, middle turns — improving the draft ───────────────────────────────

STAGE2_CLASH_INTRO = """\
Stage 2, turn {turn} — improving the draft.

Current draft:
{draft}
{critic_block}
Answer anything aimed at you. Then push the draft toward a better paper: fix a \
weakness, strengthen the claim or the way it is tested, bring in something the \
draft is missing, or — if you think it is right as it stands — say so and why. \
Silence reads as agreement, so say what you think. Substantial changes are \
welcome if they make it better research."""

SUMMARY_REQUEST = f"""\
Summarize this turn and update the draft. Return one JSON object with exactly \
two top-level keys:

{{
  "summary": {{
    "turn_recap": "1-2 sentences",
    "draft_updates": [{{"component": "what changed", "change": "how", "source": "who argued for it"}}],
    "settled": ["points everyone now accepts"],
    "open_attacks": [{{"from": "EXPERT_X", "target": "expert or draft component", "claim": "one sentence", "status": "unanswered"}}],
    "dropped": [{{"component": "what was removed", "reason": "why", "source": "who argued for it"}}],
    "unresolved": ["what is still contested"],
    "converged": false
  }},
  "updated_draft": {{schema}}
}}

Every change to the draft comes from something a researcher said this turn. \
Carry forward objections nobody answered; drop the ones that were answered. Set \
"converged" to true only when no researcher asked for further changes and the \
reviewer's earlier blocking issues are addressed.

{WRITEUP_NOTE}

Return only the JSON object."""


# ── Stage 2, final turn — final positions ─────────────────────────────────────

STAGE2_CONVERGE_INTRO = """\
Stage 2, turn {turn} — final positions.

Current draft:
{draft}
{critic_block}
Settle what is still open — anything aimed at you, and the reviewer's issues \
that touch your part of the proposal. Then give your final position: what you \
would still change, or that the draft is ready. No new directions at this point."""


# ── Stage 3 — final proposal ──────────────────────────────────────────────────

FINAL_REWRITE_REQUEST = f"""\
Stage 3 — final proposal. From the discussion and the latest draft, write the \
proposal a reader who never saw the debate would receive. Apply the changes the \
group agreed on. Where a reviewer's issue was never resolved, fix it if the \
discussion gives you the means, otherwise state it as a limitation. Keep the \
problem from Stage 1. The method or experiment must use at least one specific \
recent result the researchers brought — used, not mentioned: some step of the \
proposal depends on it being true.

Return a JSON array holding exactly one proposal object:

{{schema}}

{WRITEUP_NOTE}

Return only the JSON array."""

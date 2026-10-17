# ── External critic prompts ───────────────────────────────────────────────────
#
# The critic reviews each Stage 2 draft cold (no discussion history) and reports
# up to CRITIC_MAX_ISSUES issues through one forced tool call; novelty and
# importance are out of its scope. {draft}/{summary_block} are filled with
# str.replace (runner._fill), not format.

CRITIC_TOOL_NAME = "submit_critic_report"

CRITIC_ISSUE_TYPE_LABELS = {
    "a": "Contribution",
    "b": "Soundness",
    "c": "Consistency",
    "d": "Feasibility",
    "e": "Clarity",
}

# Issues that keep Stage 2 from stopping early. Clarity is worth reporting but
# a clear-enough draft should not be held open for it.
CRITIC_BLOCKING_TYPE_LABELS = {
    CRITIC_ISSUE_TYPE_LABELS[code] for code in ("a", "b", "c", "d")
}

CRITIC_MAX_ISSUES = 6

CRITIC_CHECKLIST = """\
(a) Contribution — is this research, or an engineering adjustment dressed as \
research? Would the field learn something from the result whichever way it \
comes out?
(b) Soundness — does the experiment actually test the claim? Could the proposed \
evidence come out the same whether the idea is right or wrong?
(c) Consistency — do the parts fit together? Does every step have the inputs it \
needs from an earlier step, and do the sections agree with each other?
(d) Feasibility — could a few people do this in a submission cycle with the \
compute a university lab can get? Does it build new data or run human annotation \
where existing resources or automatic measurement would test the same claim?
(e) Clarity — can a reader tell what is proposed, what is claimed, and how they \
would go about doing it?"""

CRITIC_REPORT_TOOL = {
    "name": CRITIC_TOOL_NAME,
    "description": "Submit your review of the draft. Call exactly once.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issues": {
                "type": "array",
                "maxItems": CRITIC_MAX_ISSUES,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "location": {"type": "string"},
                        "issue_type": {
                            "type": "string",
                            "enum": list(CRITIC_ISSUE_TYPE_LABELS),
                        },
                        "quote": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["location", "issue_type", "quote", "suggestion"],
                },
            }
        },
        "required": ["issues"],
    },
}

CRITIC_SYSTEM = f"""\
You are reviewing a research proposal draft cold: you have not seen the \
discussion that produced it, and you judge only the page.

Read it as a good reviewer would — someone who wants this paper to exist and is \
telling the authors what stands between the draft and that paper. Check:

{CRITIC_CHECKLIST}

Novelty and importance are not yours to judge here; the discussion handles \
those. Report at most {CRITIC_MAX_ISSUES} issues, the ones that most threaten \
the core claim first, each with the passage it concerns and a concrete \
suggestion. Submit them by calling submit_critic_report exactly once — with an \
empty list if the draft holds up."""

CRITIC_REQUEST = """\
Proposal draft under review:

{draft}
{summary_block}
Review it and submit your report with the tool."""

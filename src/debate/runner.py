"""
Multi-agent debate runner: the expert agents and a moderator turn the expert
system prompts into one research proposal. Stage 1 picks the problem
(statements, a three-candidate shortlist, a blind Borda vote), Stage 2 develops
a draft over DEBATE_TURNS turns with an external critic reviewing each draft,
and Stage 3 writes the final proposal. State is checkpointed after every turn
through `on_checkpoint`, so a debate resumes from its first unfinished turn.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cost_report
from config import (
    BLIND_OPENING_TURNS,
    CRITIC_ENABLED,
    CRITIC_MAX_RETRIES,
    DEBATE_TURNS,
    EARLY_STOP_ENABLED,
    FINAL_PROPOSAL_MAX_TOKENS,
    MAX_TOKENS_PER_CRITIC,
    MAX_TOKENS_PER_SPEECH,
    ORDER_MODE,
    PROBLEM_TURNS,
)
from llm_client import LLMClient
from prompts.critic_prompts import (
    CRITIC_BLOCKING_TYPE_LABELS,
    CRITIC_ISSUE_TYPE_LABELS,
    CRITIC_MAX_ISSUES,
    CRITIC_REPORT_TOOL,
    CRITIC_REQUEST,
    CRITIC_SYSTEM,
    CRITIC_TOOL_NAME,
)
from prompts.moderator_prompts import (
    DRAFT_REQUEST,
    ENDORSEMENT_SYSTEM,
    ENDORSEMENT_TOOL_NAME,
    ENDORSEMENT_VOTE_TOOL,
    FINAL_REWRITE_REQUEST,
    FINALIZE_SELECTION_NO_VOTE,
    FINALIZE_SELECTION_TIE,
    FINALIZE_SELECTION_WINNER,
    MODERATOR_SYSTEM,
    SPEECH_RULES,
    STAGE1_CANDIDATE_REQUEST,
    STAGE1_ENDORSEMENT_INTRO,
    STAGE1_FINALIZE,
    STAGE1_STATEMENT_INTRO,
    STAGE2_CLASH_INTRO,
    STAGE2_CONVERGE_INTRO,
    STAGE2_TURN1_INTRO,
    STAGE2_TURN1_VANTAGE,
    STAGE2_TURN1_VANTAGE_PAPERS,
    SUMMARY_REQUEST,
    proposal_schema,
)

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    speaker: str          # "MODERATOR" | "CRITIC" | "EXPERT_A" | ...
    stage: str            # "stage1" | "stage2" | "stage3"
    turn: int             # turn index within the stage (0 = stage-level record)
    content: str
    include_in_context: bool = True


@dataclass
class DebateResult:
    topic: str
    transcript: List[TurnRecord]
    final_ideas: List[Dict[str, Any]]
    raw_synthesis: str
    draft_versions: List[Dict[str, Any]] = field(default_factory=list)
    turn_summaries: List[Dict[str, Any]] = field(default_factory=list)
    critic_reports: List[Dict[str, Any]] = field(default_factory=list)
    early_stopped_at_turn: Optional[int] = None
    stage_reports: List[Dict[str, Any]] = field(default_factory=list)
    problem_candidates: List[Dict[str, str]] = field(default_factory=list)
    endorsement_votes: List[Dict[str, Any]] = field(default_factory=list)
    endorsement_tally: Optional[Dict[str, Any]] = None


# ── Small helpers ─────────────────────────────────────────────────────────────

def _record_from_dict(d: Dict[str, Any]) -> TurnRecord:
    return TurnRecord(
        speaker=d["speaker"],
        stage=d["stage"],
        turn=d["turn"],
        content=d["content"],
        include_in_context=d.get("include_in_context", True),
    )


def _fill(template: str, **kwargs: str) -> str:
    """Fill {name} placeholders via str.replace — several templates embed
    literal JSON braces, so str.format() is not usable."""
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", value)
    return out


# Escapes the repair pass keeps. \b and \f are legal JSON but are deliberately
# left out: no moderator ever means backspace or formfeed, whereas \bar, \beta,
# \frac and \forall are everyday LaTeX that a strict decode would silently
# mangle into control characters.
_JSON_ESCAPE_CHARS = frozenset('"\\/nrt')
_ESCAPE_PAIR_RE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)", re.S)


def _repair_json_escapes(text: str) -> str:
    r"""Double every backslash that does not start a legal JSON escape.

    Moderator output regularly carries LaTeX inside string values (\(c\),
    \frac{1}{N}, \Delta), which is not valid JSON — the whole value then fails
    to decode. Doubling turns those into literal backslashes so the text
    survives intact. Scanning escape pairs (rather than single backslashes)
    keeps already-correct "\\" and "\n" untouched."""
    def repl(match: "re.Match[str]") -> str:
        body = match.group(1)
        if len(body) == 5 and body[0] == "u":      # \uXXXX
            return match.group(0)
        if body in _JSON_ESCAPE_CHARS:
            return match.group(0)
        return "\\\\" + body
    return _ESCAPE_PAIR_RE.sub(repl, text)


def _first_json_start(text: str) -> Optional[int]:
    starts = [s for s in (text.find("{"), text.find("[")) if s != -1]
    return min(starts) if starts else None


def _decode_escaping_stray_quotes(text: str, limit: int = 100) -> Optional[Any]:
    """Decode `text`, escaping unescaped quotes inside string values as they
    are hit. A quoted phrase in prose ('binary judgment: "Does it generalize?"')
    closes the value early; the decoder then reports a missing delimiter right
    after the offending quote, which is enough to locate and escape it."""
    decoder = json.JSONDecoder()
    for _ in range(limit):
        try:
            return decoder.raw_decode(text, 0)[0]
        except json.JSONDecodeError as err:
            if "delimiter" not in err.msg:
                return None
            end = err.pos
            while end > 0 and text[end - 1].isspace():
                end -= 1
            if end == 0 or text[end - 1] != '"':
                return None
            text = text[:end - 1] + '\\"' + text[end:]
    return None


def _scan_json(text: str) -> Optional[Any]:
    """Return the first position from which a balanced JSON value decodes."""
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        start = _first_json_start(text[i:])
        if start is None:
            return None
        start += i
        try:
            obj, _ = decoder.raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            i = start + 1
    return None


def _extract_json(text: str) -> Optional[Any]:
    """Parse the first balanced JSON value ({...} or [...]) found in text.
    Handles markdown code fences and surrounding prose.

    The outermost value is tried first, then retried through the string-escaping
    repairs. Without them a single stray LaTeX escape or quoted phrase makes the
    whole proposal undecodable, and the permissive scan silently returns some
    smaller *nested* object that happens to be well-formed."""
    start = _first_json_start(text)
    if start is None:
        return None
    body = text[start:]
    decoder = json.JSONDecoder()
    repaired = _repair_json_escapes(body)
    for candidate in (body, repaired):
        try:
            return decoder.raw_decode(candidate, 0)[0]
        except json.JSONDecodeError:
            continue
    quoted = _decode_escaping_stray_quotes(repaired)
    if quoted is not None:
        return quoted
    return _scan_json(text)


def _looks_like_proposal(obj: Any) -> bool:
    """Guard against accepting some *other* well-formed object (a nested
    trajectory_attribution block, say) as the proposal."""
    return (
        isinstance(obj, dict)
        and all(isinstance(obj.get(k), str) and obj[k].strip()
                for k in ("title", "method"))
    )


# The shortlist is voted on blind, so a candidate that names the expert it came
# from defeats the whole mechanism. Detection is deliberately broad (it only
# triggers a regeneration), while the last-resort scrub is narrow — it must not
# mangle legitimate prose such as "domain experts hand-engineer the rubric" or
# "an expert a priori bound".
_EXPERT_REF_DETECT_RE = re.compile(
    r"\bexperts?[_ ]?[A-F]\b"
    r"|\b(?:the|one|another|each|that|this)\s+experts?(?:'s)?\b"
    r"|\b(?:first|second|third)\s+experts?\b"
    r"|\bexperts?'s\s+(?:trajectory|lineage|statement|tension)\b",
    re.IGNORECASE,
)
_EXPERT_REF_SCRUB_RE = re.compile(r"\b(?:Expert|EXPERT)[_ ]?[A-F]\b")


def _candidate_attribution_leak(candidates: List[Dict[str, str]]) -> Optional[str]:
    """First participant reference found in the shortlist, or None if clean."""
    for index, candidate in enumerate(candidates, 1):
        for field_name in ("problem", "why_significant"):
            match = _EXPERT_REF_DETECT_RE.search(candidate[field_name])
            if match:
                return f"candidate {index} {field_name}: {match.group(0)!r}"
    return None


def _scrub_candidate_attribution(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Last-resort blinding when the moderator keeps naming experts after its
    retries. Only the unambiguous `Expert A` / `EXPERT_B` form is rewritten;
    softer references are left alone rather than risk corrupting the problem
    statement, and the caller logs that the shortlist is imperfectly blinded."""
    return [
        {
            field_name: _EXPERT_REF_SCRUB_RE.sub("an expert", value)
            for field_name, value in candidate.items()
        }
        for candidate in candidates
    ]


def _shuffle_candidates(candidates: List[Dict[str, str]], topic: str) -> List[Dict[str, str]]:
    """Break the `candidate #1 == first speaker` correlation. Seeded from the
    topic so a resumed or re-run debate produces the same shortlist order."""
    seed = int(hashlib.sha256(topic.encode("utf-8")).hexdigest()[:16], 16)
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _parse_problem_candidates(text: str) -> Optional[List[Dict[str, str]]]:
    """Return a validated three-problem shortlist, or None when malformed."""
    parsed = _extract_json(text)
    if not isinstance(parsed, dict) or set(parsed) != {"candidates"}:
        return None
    candidates = parsed["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        return None
    required_fields = {"problem", "why_significant"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != required_fields:
            return None
        if any(
            not isinstance(candidate[field], str) or not candidate[field].strip()
            for field in required_fields
        ):
            return None
    return candidates


def _format_candidates(candidates: List[Dict[str, str]]) -> str:
    """Shortlist as the experts see it in the blind endorsement vote."""
    return "\n\n".join(
        f"Candidate {index}\n"
        f"  problem: {candidate['problem']}\n"
        f"  why_significant: {candidate['why_significant']}"
        for index, candidate in enumerate(candidates, 1)
    )


_ENDORSEMENT_RANKING_FIELDS = {"candidate", "rank", "reason"}
_ENDORSEMENT_MAX_RETRIES = 2          # 3 attempts in total
_MAX_TOKENS_PER_VOTE = 800            # ranking + three one-or-two-sentence reasons


def _validate_endorsement_ranking(
    payload: Any, n_candidates: int = 3,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Validate the forced endorsement payload: `candidate` and `rank` must each
    be a permutation of 1..n_candidates, every reason non-empty. Returned in
    rank order so the record reads top choice first."""
    if not isinstance(payload, dict):
        return None, "endorsement payload must be an object"
    if set(payload) != {"ranking"}:
        return None, "endorsement payload must contain exactly the top-level key 'ranking'"

    ranking = payload["ranking"]
    if not isinstance(ranking, list) or len(ranking) != n_candidates:
        return None, f"endorsement 'ranking' must be a list of exactly {n_candidates} entries"

    expected = set(range(1, n_candidates + 1))
    for index, entry in enumerate(ranking):
        if not isinstance(entry, dict) or set(entry) != _ENDORSEMENT_RANKING_FIELDS:
            return None, (
                f"ranking entry {index} must contain exactly "
                f"{sorted(_ENDORSEMENT_RANKING_FIELDS)}"
            )
        for field_name in ("candidate", "rank"):
            # bool is an int subclass; a JSON `true` must not pass as rank 1.
            if not isinstance(entry[field_name], int) or isinstance(entry[field_name], bool):
                return None, f"ranking entry {index} '{field_name}' must be an integer"
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            return None, f"ranking entry {index} 'reason' must be a non-empty string"

    if {e["candidate"] for e in ranking} != expected:
        return None, f"'candidate' values must be each of {sorted(expected)} exactly once"
    if {e["rank"] for e in ranking} != expected:
        return None, f"'rank' values must be each of {sorted(expected)} exactly once"

    return sorted(ranking, key=lambda e: e["rank"]), None


def _tally_rankings(
    votes: List[Dict[str, Any]], n_candidates: int = 3,
) -> Dict[str, Any]:
    """Borda tally over the endorsement rankings: a candidate scores
    `n_candidates - rank + 1` points from each voter (top choice = 3 points with
    the default shortlist of 3). `winner` holds every candidate on the top score
    — one entry means the vote decided, more than one means the moderator gets a
    tie to break."""
    points = {c: 0 for c in range(1, n_candidates + 1)}
    first_choice = {c: 0 for c in range(1, n_candidates + 1)}
    for vote in votes:
        for entry in vote["ranking"]:
            points[entry["candidate"]] += n_candidates - entry["rank"] + 1
            if entry["rank"] == 1:
                first_choice[entry["candidate"]] += 1
    top = max(points.values()) if points else 0
    return {
        "method": f"borda(top={n_candidates})",
        "voters": [vote["expert"] for vote in votes],
        "points": {str(c): points[c] for c in sorted(points)},
        "first_choice_votes": {str(c): first_choice[c] for c in sorted(first_choice)},
        "winner": [c for c in sorted(points) if points[c] == top],
    }


_CRITIC_ISSUE_FIELDS = {"location", "issue_type", "quote", "suggestion"}
_CRITIC_ISSUE_TYPES = set(CRITIC_ISSUE_TYPE_LABELS)
_CRITIC_MAX_ISSUES = CRITIC_MAX_ISSUES


def _validate_critic_report(payload: Any) -> tuple[Optional[List[Dict]], Optional[str]]:
    """Validate the forced critic tool payload without permissive JSON recovery."""
    if not isinstance(payload, dict):
        return None, "critic payload must be an object"
    if set(payload) != {"issues"}:
        return None, "critic payload must contain exactly the top-level key 'issues'"

    issues = payload["issues"]
    if not isinstance(issues, list):
        return None, "critic payload 'issues' must be a list"
    if len(issues) > _CRITIC_MAX_ISSUES:
        return None, f"critic payload contains {len(issues)} issues; maximum is {_CRITIC_MAX_ISSUES}"

    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            return None, f"critic issue {index} must be an object"
        if set(issue) != _CRITIC_ISSUE_FIELDS:
            return None, (
                f"critic issue {index} must contain exactly "
                f"{sorted(_CRITIC_ISSUE_FIELDS)}"
            )
        if issue["issue_type"] not in _CRITIC_ISSUE_TYPES:
            return None, f"critic issue {index} has invalid issue_type {issue['issue_type']!r}"
        for field_name in ("location", "quote", "suggestion"):
            value = issue[field_name]
            if not isinstance(value, str) or not value.strip():
                return None, f"critic issue {index}.{field_name} must be a non-empty string"

    return issues, None


def _describe_critic_issues(issues: List[Dict]) -> List[Dict]:
    """Replace compact critic type codes with labels before storage/display."""
    return [
        {**issue, "issue_type": CRITIC_ISSUE_TYPE_LABELS[issue["issue_type"]]}
        for issue in issues
    ]


def _blocking_critic_issues(issues: List[Dict]) -> List[Dict]:
    return [
        issue for issue in issues
        if issue.get("issue_type") in CRITIC_BLOCKING_TYPE_LABELS
    ]


def _format_transcript(transcript: List[TurnRecord]) -> str:
    records = [record for record in transcript if record.include_in_context]
    if not records:
        return "(No messages yet.)"
    return "\n\n".join(f"[{r.speaker}]: {r.content}" for r in records)


_STAGE2_TURN_LABELS = {
    "propose": "Propose (method eval + initial draft + critic)",
    "clash": "Clash (respond/engage + turn summary + critic)",
    "converge": "Converge (final positions, no new attacks)",
}
_STAGE2_TURN_EXPECTED_CALL_TYPES = {
    "propose": ["expert_speech", "moderator", "critic"],
    "clash": ["order", "expert_speech", "moderator", "critic"],
    "converge": ["order", "expert_speech"],
}


def _stage2_turn_entry(
    turn: int, turn_type: str, calls: List[Dict], duration: float, model: str, *,
    converged: Optional[bool] = None, skipped: bool = False, skip_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return cost_report.make_stage_entry(
        stage_id=f"debate.stage2.turn{turn}_{turn_type}",
        label=f"Debate Stage 2 Turn {turn} — {_STAGE2_TURN_LABELS[turn_type]}",
        calls=calls, duration_sec=duration, model=model,
        converged=converged,
        skipped=skipped, skip_reason=skip_reason,
        expected_call_types=_STAGE2_TURN_EXPECTED_CALL_TYPES[turn_type],
    )


_STAGE1_TURN_LABELS = {
    "statement": "Statement (initial tension per expert)",
    "endorsement": "Endorsement (blind ranking vote)",
}


def _stage1_turn_entry(
    turn: int, turn_type: str, calls: List[Dict], duration: float, model: str, *,
    finalize: bool = False, candidate_synthesis: bool = False,
) -> Dict[str, Any]:
    """`finalize=True` on the last stage-1 turn folds in the moderator's
    problem write-up call, which runs once at the end of stage 1 rather than
    per turn. `candidate_synthesis=True` on turn 1 folds in the moderator's
    shortlist call."""
    label = f"Debate Stage 1 Turn {turn} — {_STAGE1_TURN_LABELS[turn_type]}"
    # The endorsement turn is a forced tool call per expert, not a speech.
    expected_call_types = ["expert_speech" if turn_type == "statement" else "vote"]
    if candidate_synthesis:
        label += " + candidate synthesis (three problems)"
        expected_call_types.append("moderator")
    if finalize:
        label += " + finalize (problem statement)"
        expected_call_types.append("moderator")
    return cost_report.make_stage_entry(
        stage_id=f"debate.stage1.turn{turn}_{turn_type}",
        label=label,
        calls=calls, duration_sec=duration, model=model,
        expected_call_types=expected_call_types,
    )


def _run_expert_speech(
    expert_name: str,
    system_prompt: str,
    instruction: str,
    transcript: List[TurnRecord],
    stage: str,
    turn: int,
    llm_client: LLMClient,
    cost_tag: str,
    blind: bool = False,
) -> TurnRecord:
    """
    One expert speech: the expert sees prior substantive debate records + its
    current instruction. `cost_tag` is the cost-tracking tag for the call.
    """
    # `blind`: the expert sees only its own instruction, never the transcript, so
    # an opening turn is written independently instead of anchored on whoever
    # spoke first (config.BLIND_OPENING_TURNS). The speech is still appended to
    # the transcript, so every later turn sees all of them.
    if blind:
        user_content = f"[MODERATOR → {expert_name}]: {instruction}"
    else:
        context = _format_transcript(transcript)
        user_content = f"{context}\n\n---\n[MODERATOR → {expert_name}]: {instruction}"

    response = llm_client.complete(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=MAX_TOKENS_PER_SPEECH,
        stage=cost_tag,
    )
    final_text = response.text or ""
    if not final_text.strip():
        # Not fatal — the moderator can still run — but it silently degrades every
        # downstream stage, so it must never pass unnoticed.
        logger.warning("%s produced an empty speech (stage=%s turn=%s)", expert_name, stage, turn)

    return TurnRecord(
        speaker=expert_name,
        stage=stage,
        turn=turn,
        content=final_text,
    )


def _run_moderator_call(
    instruction: str,
    transcript: List[TurnRecord],
    stage: str,
    turn: int,
    llm_client: LLMClient,
    cost_tag: str,
    max_tokens: int = 2000,
    system_prompt: str = MODERATOR_SYSTEM,
) -> TurnRecord:
    """One moderator call over substantive prior debate records."""
    context = _format_transcript(transcript)
    user_content = f"{context}\n\n---\n[INSTRUCTION]: {instruction}"

    response = llm_client.complete(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        tools=None,
        max_tokens=max_tokens,
        stage=cost_tag,
    )
    return TurnRecord(
        speaker="MODERATOR",
        stage=stage,
        turn=turn,
        content=response.text or "",
    )


# ── Main debate runner ────────────────────────────────────────────────────────

class DebateRunner:
    def __init__(
        self,
        topic: str,
        expert_system_prompts: List[str],   # length == K
        llm_client: LLMClient,
        checkpoint: Optional[Dict[str, Any]] = None,
        on_checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.topic   = topic
        self.experts = expert_system_prompts
        self.k       = len(expert_system_prompts)
        self.names   = [f"EXPERT_{chr(65 + i)}" for i in range(self.k)]
        self.llm     = llm_client
        self.transcript: List[TurnRecord] = []
        self.schema  = proposal_schema(self.names)

        # Global rotation counter: each expert turn (across stages) starts one
        # position later, so leading the discussion rotates through all experts.
        self._rotation_counter = 0

        # Metric/state accumulators
        self.draft_versions: List[Dict[str, Any]] = []
        self.turn_summaries: List[Dict[str, Any]] = []
        self.critic_reports: List[Dict[str, Any]] = []
        self.early_stopped_at_turn: Optional[int] = None

        # Stage 1 endorsement vote (blind ranking → Borda tally → finalise).
        self._candidates: Optional[List[Dict[str, str]]] = None
        self.endorsement_votes: List[Dict[str, Any]] = []
        self.endorsement_tally: Optional[Dict[str, Any]] = None

        # Cost-report stage entries
        # one per debate stage/turn, in execution order — see cost_report.py.
        self.stage_reports: List[Dict[str, Any]] = []

        # Turn-level resume state. m/n are frozen per debate (a checkpoint's
        # values win over config, so a resumed debate keeps its original shape).
        self._m = max(1, PROBLEM_TURNS)
        self._n = DEBATE_TURNS
        if self._n < 3:
            logger.warning("DEBATE_TURNS=%d below the minimum of 3; using 3.", self._n)
            self._n = 3
        self._stage1_turns_done = 0
        self._stage2_turns_done = 0
        self._stage2_finished = False
        self._draft: Optional[Dict] = None
        self._open_attacks: List[Dict] = []
        self._critic_issues: List[Dict] = []

        self._on_checkpoint = on_checkpoint
        if checkpoint:
            self._restore(checkpoint)

    # ── checkpoint / resume ───────────────────────────────────────────────────

    def checkpoint(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of everything needed to resume the debate
        after the last fully completed turn (stage 3 is the final step, so it
        needs no checkpoint of its own — finishing it completes the debate).
        Deep-copied: the caller may hold the snapshot across later turns (e.g.
        to persist it in a crash handler), so it must not alias the live
        accumulator lists this runner keeps mutating."""
        return copy.deepcopy({
            "k": self.k,
            "m": self._m,
            "n": self._n,
            "rotation_counter": self._rotation_counter,
            "transcript": [asdict(r) for r in self.transcript],
            "draft_versions": self.draft_versions,
            "turn_summaries": self.turn_summaries,
            "critic_reports": self.critic_reports,
            "problem_candidates": self._candidates,
            "endorsement_votes": self.endorsement_votes,
            "endorsement_tally": self.endorsement_tally,
            "stage_reports": self.stage_reports,
            "early_stopped_at_turn": self.early_stopped_at_turn,
            "stage1_turns_done": self._stage1_turns_done,
            "stage2_turns_done": self._stage2_turns_done,
            "stage2_finished": self._stage2_finished,
            "draft": self._draft,
            "open_attacks": self._open_attacks,
            "critic_issues": self._critic_issues,
        })

    def _restore(self, cp: Dict[str, Any]) -> None:
        # Deep-copy for the same aliasing reason as checkpoint(): the caller
        # keeps holding cp (e.g. inside its progress dict) while this runner
        # mutates the restored lists.
        cp = copy.deepcopy(cp)
        if cp.get("k") != self.k:
            raise ValueError(
                f"Debate checkpoint was made with k={cp.get('k')} experts but "
                f"{self.k} expert prompts were provided."
            )
        if cp["m"] != self._m or cp["n"] != self._n:
            logger.warning(
                "Debate checkpoint was started with m=%d, n=%d; current config says "
                "m=%d, n=%d — keeping the checkpoint's values so the debate stays "
                "internally consistent.",
                cp["m"], cp["n"], self._m, self._n,
            )
            self._m = cp["m"]
            self._n = cp["n"]
        self._rotation_counter = cp["rotation_counter"]
        self.transcript = [_record_from_dict(d) for d in cp["transcript"]]
        self.draft_versions = cp["draft_versions"]
        self.turn_summaries = cp["turn_summaries"]
        self.critic_reports = cp["critic_reports"]
        # .get for all three: a checkpoint without the endorsement-vote fields
        # still resumes (the shortlist is recovered from the transcript below).
        self._candidates = cp.get("problem_candidates")
        self.endorsement_votes = cp.get("endorsement_votes", [])
        self.endorsement_tally = cp.get("endorsement_tally")
        self.stage_reports = cp["stage_reports"]
        self.early_stopped_at_turn = cp["early_stopped_at_turn"]
        self._stage1_turns_done = cp["stage1_turns_done"]
        self._stage2_turns_done = cp["stage2_turns_done"]
        self._stage2_finished = cp["stage2_finished"]
        self._draft = cp["draft"]
        self._open_attacks = cp["open_attacks"]
        self._critic_issues = cp["critic_issues"]
        if self._candidates is None and self._stage1_turns_done >= 1 and self._m > 1:
            # Checkpoint carries no stored shortlist: it is still in the
            # transcript, so recover it instead of failing the resume. It is
            # NOT re-shuffled — the transcript record the moderator reads at
            # finalise already numbers the candidates in that order, and a
            # renumbered ballot would not line up with it.
            recovered = next(
                (c for c in (
                    _parse_problem_candidates(r.content)
                    for r in reversed(self.transcript)
                    if r.stage == "stage1" and r.turn == 1 and r.speaker == "MODERATOR"
                ) if c is not None),
                None,
            )
            if recovered is not None:
                self._candidates = recovered
                logger.warning(
                    "Recovered the stage-1 shortlist from a checkpoint without a stored one; it "
                    "keeps its original order, so this resumed topic's ballot is not "
                    "shuffled."
                )
        logger.info(
            "Debate checkpoint restored: stage1 %d/%d turn(s), stage2 %d/%d turn(s)%s.",
            self._stage1_turns_done, self._m, self._stage2_turns_done, self._n,
            " (stage 2 finished)" if self._stage2_finished else "",
        )

    def _save_checkpoint(self) -> None:
        if self._on_checkpoint is not None:
            self._on_checkpoint(self.checkpoint())

    # ── generic helpers ───────────────────────────────────────────────────────

    def _add(self, record: TurnRecord) -> None:
        self.transcript.append(record)
        logger.info("[%s turn %d] %s: %s…",
                    record.stage, record.turn, record.speaker, record.content[:120])

    def _prompt_of(self, name: str) -> str:
        return self.experts[self.names.index(name)]

    def _rotation(self) -> List[str]:
        start = self._rotation_counter % self.k
        self._rotation_counter += 1
        return [self.names[(start + i) % self.k] for i in range(self.k)]

    def _speaking_order(self, open_attacks: List[Dict], cost_tag: str) -> List[str]:
        """Clash-turn speaking order. Default rule: experts with unanswered
        attacks against them speak first (they owe responses), remainder in
        rotation. ORDER_MODE='llm' lets the moderator pick.

        `cost_tag` (e.g. "stage2_clash.order") tags the LLM call made when
        ORDER_MODE='llm'; unused otherwise."""
        rotation = self._rotation()

        if ORDER_MODE == "llm":
            order = self._llm_order(open_attacks, cost_tag)
            if order:
                return order
            logger.warning("ORDER_MODE=llm returned no valid order; falling back to rule.")

        priority: List[str] = []
        for atk in open_attacks:
            if str(atk.get("status", "unanswered")).lower() != "unanswered":
                continue
            target = str(atk.get("target", ""))
            for nm in self.names:
                if nm in target and nm not in priority:
                    priority.append(nm)
        rest = [nm for nm in rotation if nm not in priority]
        order = priority + rest
        if priority:
            logger.info("Speaking order %s (attack-priority: %s)", order, priority)
        else:
            logger.info("Speaking order %s (rotation, no open attacks)", order)
        return order

    def _llm_order(self, open_attacks: List[Dict], cost_tag: str) -> Optional[List[str]]:
        instruction = (
            "Choose the speaking order for the experts this turn. Experts with "
            "unanswered attacks against them should generally speak early (they "
            f"owe responses). Experts: {', '.join(self.names)}. Open attacks: "
            f"{json.dumps(open_attacks, ensure_ascii=False)}. "
            "Output ONLY the comma-separated expert names in speaking order."
        )
        response = self.llm.complete(
            system_prompt=MODERATOR_SYSTEM,
            messages=[{"role": "user", "content": instruction}],
            tools=None,
            max_tokens=100,
            stage=cost_tag,
        )
        raw = (response.text or "").strip()
        picked = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        if sorted(picked) == sorted(self.names):
            return picked
        return None

    def _attacks_on(self, name: str, open_attacks: List[Dict]) -> List[Dict]:
        return [
            a for a in open_attacks
            if str(a.get("status", "unanswered")).lower() == "unanswered"
            and name in str(a.get("target", ""))
        ]

    def _attack_block(self, name: str, open_attacks: List[Dict]) -> str:
        mine = self._attacks_on(name, open_attacks)
        if not mine:
            return "Nothing is aimed at you this turn."
        lines = [
            f"{i}. [{a.get('from', '?')}] on «{a.get('target', '?')}»: {a.get('claim', '?')}"
            for i, a in enumerate(mine, 1)
        ]
        return "Aimed at you this turn — answer each:\n" + "\n".join(lines)

    def _critic_block(self, issues: List[Dict]) -> str:
        if not issues:
            return "\n"
        lines = [
            f"- ({i.get('issue_type', '?')}) {i.get('location', '?')}: "
            f"{i.get('suggestion', i.get('quote', '?'))}"
            for i in issues
        ]
        return (
            "\nReviewer notes on the current draft (the reviewer has not seen the "
            "discussion):\n" + "\n".join(lines) + "\n"
        )

    def _run_critic(self, draft: Optional[Dict], summary: Optional[Dict],
                    turn: int, cost_tag: str) -> List[Dict]:
        """External critic: sees ONLY the current draft (+ turn summary), never
        the transcript. Uses a forced structured tool response and fails closed
        after retrying malformed or truncated reports."""
        if not CRITIC_ENABLED or draft is None:
            return []
        summary_block = ""
        if summary is not None:
            summary_block = ("\nModerator's turn summary:\n"
                             + json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        base_user_content = _fill(
            CRITIC_REQUEST,
            draft=json.dumps(draft, ensure_ascii=False, indent=2),
            summary_block=summary_block,
        )
        last_error = "critic returned no response"

        for attempt in range(1, CRITIC_MAX_RETRIES + 2):
            retry_note = ""
            if attempt > 1:
                retry_note = (
                    "\n\nThe previous response was malformed or truncated. "
                    f"Submit one concise tool call with at most {CRITIC_MAX_ISSUES} issues."
                )
            try:
                response = self.llm.complete(
                    system_prompt=CRITIC_SYSTEM,
                    messages=[{"role": "user", "content": base_user_content + retry_note}],
                    tools=[CRITIC_REPORT_TOOL],
                    max_tokens=MAX_TOKENS_PER_CRITIC,
                    required_tool_name=CRITIC_TOOL_NAME,
                    stage=cost_tag,
                )
            except json.JSONDecodeError as exc:
                last_error = f"critic tool arguments were invalid JSON: {exc}"
            else:
                matching_calls = [
                    call for call in response.tool_calls
                    if call.name == CRITIC_TOOL_NAME
                ]
                if len(response.tool_calls) != 1 or len(matching_calls) != 1:
                    last_error = (
                        "critic must call submit_critic_report exactly once; "
                        f"received {[call.name for call in response.tool_calls]}"
                    )
                else:
                    coded_issues, validation_error = _validate_critic_report(
                        matching_calls[0].input,
                    )
                    if validation_error is None and coded_issues is not None:
                        issues = _describe_critic_issues(coded_issues)
                        canonical = json.dumps(
                            {"issues": issues}, ensure_ascii=False, indent=2,
                        )
                        self._add(TurnRecord(
                            speaker="CRITIC", stage="stage2", turn=turn,
                            content=canonical,
                        ))
                        self.critic_reports.append({"turn": turn, "issues": issues})
                        if issues:
                            logger.info(
                                "Critic (after stage2 turn %d): %d issue(s) — types: %s",
                                turn, len(issues),
                                ",".join(sorted({str(i["issue_type"]) for i in issues})),
                            )
                        else:
                            logger.info("Critic (after stage2 turn %d): no issues.", turn)
                        return issues
                    last_error = validation_error or "unknown critic validation error"

            logger.warning(
                "Critic response invalid at stage2 turn %d (attempt %d/%d): %s",
                turn, attempt, CRITIC_MAX_RETRIES + 1, last_error,
            )

        raise RuntimeError(
            f"Critic response invalid after {CRITIC_MAX_RETRIES + 1} attempt(s) "
            f"at stage2 turn {turn}: {last_error}"
        )

    def _synthesise_candidates(self, turn: int) -> List[Dict[str, str]]:
        """Moderator's three-problem shortlist, blinded and shuffled.

        Blinding is what makes the endorsement vote meaningful, so a shortlist
        that names its experts is regenerated rather than accepted; only after
        the retries are spent does the narrow scrub run, and the shortfall is
        logged rather than swallowed."""
        candidate_request = STAGE1_CANDIDATE_REQUEST
        candidate_record: Optional[TurnRecord] = None
        candidates: Optional[List[Dict[str, str]]] = None
        leak: Optional[str] = None

        for attempt in range(1, 4):
            response = _run_moderator_call(
                candidate_request, self.transcript, "stage1", turn, self.llm,
                cost_tag="stage1_statement.candidate_synthesis", max_tokens=2000,
            )
            parsed = _parse_problem_candidates(response.content)
            if parsed is None:
                logger.warning("Stage 1 candidate synthesis malformed (attempt %d/3).", attempt)
                candidate_request = (
                    STAGE1_CANDIDATE_REQUEST
                    + "\n\nYour previous response was malformed. Return exactly three "
                    "candidates with only problem and why_significant."
                )
                continue

            leak = _candidate_attribution_leak(parsed)
            if leak is None:
                candidates, candidate_record = parsed, response
                break

            logger.warning(
                "Stage 1 shortlist names a participant (attempt %d/3) — %s.", attempt, leak,
            )
            candidates, candidate_record = parsed, response
            candidate_request = (
                STAGE1_CANDIDATE_REQUEST
                + f"\n\nYour previous response broke the anonymity rule ({leak}). "
                "Rewrite all three candidates with every reference to a discussion "
                "participant removed."
            )

        if candidates is None or candidate_record is None:
            raise RuntimeError(
                "Stage 1 candidate synthesis failed to return three valid candidates "
                "after 3 attempts."
            )
        if leak is not None:
            logger.warning(
                "Stage 1 shortlist still referenced a participant after 3 attempts; "
                "scrubbing the explicit forms. The vote is blinded imperfectly for "
                "this topic (%s).", self.topic,
            )
            candidates = _scrub_candidate_attribution(candidates)

        candidates = _shuffle_candidates(candidates, self.topic)
        candidate_record.content = json.dumps(
            {"candidates": candidates}, ensure_ascii=False, indent=2,
        )
        self._add(candidate_record)
        return candidates

    def _run_endorsement_vote(self, name: str, turn: int) -> Dict[str, Any]:
        """One expert's blind ranking of the shortlist.

        The expert sees ONLY the topic and the anonymised shortlist — never the
        statement turn, never the other experts' votes. That is deliberate: with
        the full transcript it could identify its own candidate from its own
        turn-1 wording, and (speaking last in the rotation) anchor on whoever
        voted before it. Both are exactly the biases this vote exists to remove,
        so the context is rebuilt from scratch instead of reusing the
        transcript.
        """
        user_content = _fill(
            STAGE1_ENDORSEMENT_INTRO,
            turn=str(turn),
            topic=self.topic,
            candidates=_format_candidates(self._candidates or []),
        )
        last_error = "endorsement returned no response"

        for attempt in range(1, _ENDORSEMENT_MAX_RETRIES + 2):
            retry_note = ""
            if attempt > 1:
                retry_note = (
                    f"\n\nYour previous response was rejected ({last_error}). Call "
                    "submit_endorsement exactly once, ranking every candidate."
                )
            try:
                response = self.llm.complete(
                    system_prompt=ENDORSEMENT_SYSTEM,
                    messages=[{"role": "user", "content": user_content + retry_note}],
                    tools=[ENDORSEMENT_VOTE_TOOL],
                    max_tokens=_MAX_TOKENS_PER_VOTE,
                    required_tool_name=ENDORSEMENT_TOOL_NAME,
                    stage="stage1_endorsement.vote",
                )
            except json.JSONDecodeError as exc:
                last_error = f"tool arguments were invalid JSON: {exc}"
                continue

            matching = [c for c in response.tool_calls if c.name == ENDORSEMENT_TOOL_NAME]
            if len(response.tool_calls) != 1 or len(matching) != 1:
                last_error = (
                    "must call submit_endorsement exactly once; received "
                    f"{[c.name for c in response.tool_calls]}"
                )
                continue

            ranking, error = _validate_endorsement_ranking(matching[0].input)
            if error is not None or ranking is None:
                last_error = error or "unknown endorsement validation error"
                logger.warning("%s endorsement invalid (attempt %d/%d): %s",
                               name, attempt, _ENDORSEMENT_MAX_RETRIES + 1, last_error)
                continue

            vote = {"turn": turn, "expert": name, "ranking": ranking}
            self._add(TurnRecord(
                speaker=name, stage="stage1", turn=turn,
                content=json.dumps({"ranking": ranking}, ensure_ascii=False, indent=2),
            ))
            return vote

        raise RuntimeError(
            f"{name} failed to submit a valid endorsement ranking after "
            f"{_ENDORSEMENT_MAX_RETRIES + 1} attempt(s): {last_error}"
        )

    def _finalize_request(self, votes: List[Dict[str, Any]]) -> str:
        """Turn the tally into the moderator's finalise instruction. A single
        top-scoring candidate leaves the moderator nothing to choose — it writes
        the winner up. Only a tie hands the choice back, and only within the
        tied set."""
        candidates = self._candidates or []
        tally = _tally_rankings(votes, n_candidates=len(candidates))
        self.endorsement_tally = tally
        self._add(TurnRecord(
            speaker="MODERATOR", stage="stage1", turn=self._m,
            content=json.dumps({"endorsement_tally": tally}, ensure_ascii=False, indent=2),
        ))
        winners = tally["winner"]
        logger.info(
            "Stage 1 endorsement tally: points=%s, winner=%s%s",
            tally["points"], winners, " (tie — moderator breaks it)" if len(winners) > 1 else "",
        )
        winner_block = "\n\n".join(
            f"Candidate {c}\n"
            f"  problem: {candidates[c - 1]['problem']}\n"
            f"  why_significant: {candidates[c - 1]['why_significant']}"
            for c in winners
        )
        selection = _fill(
            FINALIZE_SELECTION_WINNER if len(winners) == 1 else FINALIZE_SELECTION_TIE,
            winner=winner_block,
            tally=json.dumps(tally, ensure_ascii=False, indent=2),
        )
        return _fill(STAGE1_FINALIZE, selection=selection)

    # ── Stage 1 — problem determination ──────────────────────────────────────

    def _stage1(self) -> None:
        t0 = time.monotonic()
        m = self._m
        start = self._stage1_turns_done + 1
        logger.info("── Stage 1 (problem determination, %d turn(s)) started%s.",
                    m, f" from turn {start} (resumed)" if start > 1 else "")
        synth: Optional[TurnRecord] = None

        for t in range(start, m + 1):
            call_snap = self.llm.call_log_snapshot()
            t_turn = time.monotonic()
            turn_type = "statement" if t == 1 else "endorsement"

            if t == 1:
                intro = _fill(STAGE1_STATEMENT_INTRO, topic=self.topic)
                instruction = SPEECH_RULES
                self._add(TurnRecord(
                    speaker="MODERATOR", stage="stage1", turn=t, content=intro,
                    include_in_context=False,
                ))
                for name in self._rotation():
                    self._add(_run_expert_speech(
                        name, self._prompt_of(name), f"{intro}\n\n{instruction}",
                        self.transcript, "stage1", t, self.llm,
                        cost_tag="stage1_statement.expert_speech",
                        blind=BLIND_OPENING_TURNS,
                    ))
                if m > 1:
                    self._candidates = self._synthesise_candidates(t)
            else:
                if not self._candidates:
                    raise RuntimeError(
                        "Stage 1 endorsement turn reached without a candidate shortlist."
                    )
                # Recorded for the transcript log only (kept out of every later
                # call's context); the experts are handed this text directly
                # rather than reading it out of the transcript (see
                # _run_endorsement_vote).
                self._add(TurnRecord(
                    speaker="MODERATOR", stage="stage1", turn=t,
                    content=_fill(
                        STAGE1_ENDORSEMENT_INTRO,
                        turn=str(t),
                        topic=self.topic,
                        candidates=_format_candidates(self._candidates),
                    ),
                    include_in_context=False,
                ))
                # The rotation is still consumed so later stages keep their
                # speaking order, but it does not affect the outcome: the
                # votes are independent, so vote order carries no information.
                # With m > 2 each extra turn re-runs the identical blind ballot
                # and only the last turn's votes are tallied — m = 2 is the
                # designed shape; the accumulator keeps every round for analysis.
                votes = [self._run_endorsement_vote(name, t) for name in self._rotation()]
                # A crash between the votes and the finalise call resumes at this
                # same turn, so drop any votes it already recorded for it.
                self.endorsement_votes = [
                    v for v in self.endorsement_votes if v["turn"] != t
                ] + votes

            if t == m:
                finalize_req = (_fill(STAGE1_FINALIZE, selection=FINALIZE_SELECTION_NO_VOTE)
                                if m == 1 else self._finalize_request(votes))
                synth = _run_moderator_call(finalize_req, self.transcript,
                                            "stage1", t, self.llm,
                                            cost_tag=f"stage1_{turn_type}.moderator")
                self._add(synth)

            self.stage_reports.append(_stage1_turn_entry(
                t, turn_type, self.llm.calls_since(call_snap), time.monotonic() - t_turn,
                self.llm.model, finalize=(t == m),
                candidate_synthesis=(t == 1 and m > 1),
            ))
            self._stage1_turns_done = t
            self._save_checkpoint()

        duration = time.monotonic() - t0
        logger.info("── Stage 1 done in %.1fs — problem statement finalised (%d chars).",
                    duration, len(synth.content) if synth else 0)

    # ── Stage 2 — debate turns ────────────────────────────────────────────────

    def _stage2(self) -> None:
        t0 = time.monotonic()
        n = self._n
        start = self._stage2_turns_done + 1
        logger.info("── Stage 2 (debate, %d turns planned) started%s.",
                    n, f" from turn {start} (resumed)" if start > 1 else "")

        # A checkpoint taken right at an early stop has early_stopped_at_turn
        # set but the skipped entries not yet appended — resuming from it must
        # not run the remaining turns, only fall through to append them.
        if self.early_stopped_at_turn is None:
            for turn in range(start, n + 1):
                call_snap = self.llm.call_log_snapshot()
                t_turn = time.monotonic()
                turn_type = "propose" if turn == 1 else ("converge" if turn == n else "clash")
                converged_flag: Optional[bool] = None
                logger.info("── Stage 2 turn %d/%d (%s) started — %d open attack(s), "
                            "%d critic issue(s) pending.",
                            turn, n, turn_type, len(self._open_attacks), len(self._critic_issues))
                if turn == 1:
                    self._draft = self._propose_turn()
                    self._critic_issues = self._run_critic(
                        self._draft, None, turn, cost_tag="stage2_propose.critic")
                elif turn < n:
                    self._draft, summary = self._clash_turn(
                        turn, self._draft, self._open_attacks, self._critic_issues)
                    if summary:
                        self._open_attacks = summary.get("open_attacks", [])
                    self._critic_issues = self._run_critic(
                        self._draft, summary, turn, cost_tag="stage2_clash.critic")
                    converged_flag = bool(summary.get("converged")) if summary else False
                else:
                    self._converge_turn(turn, self._draft, self._open_attacks, self._critic_issues)

                turn_duration = time.monotonic() - t_turn
                logger.info("── Stage 2 turn %d done in %.1fs.", turn, turn_duration)
                self.stage_reports.append(_stage2_turn_entry(
                    turn, turn_type,
                    self.llm.calls_since(call_snap), turn_duration, self.llm.model,
                    converged=(converged_flag if turn_type == "clash" else None),
                ))
                self._stage2_turns_done = turn

                if turn_type == "clash" and EARLY_STOP_ENABLED and converged_flag:
                    blocking = _blocking_critic_issues(self._critic_issues)
                    if blocking:
                        logger.info(
                            "Early stop denied at stage2 turn %d: current critic "
                            "reported %d blocking issue(s).",
                            turn, len(blocking),
                        )
                    else:
                        # The moderator confirmed agreement and closure of the
                        # previous report; the current cold review found no new
                        # blocking issue, so this turn's endorsements can stand
                        # as the final positions.
                        logger.info("Early stop: converged at stage2 turn %d.", turn)
                        self.early_stopped_at_turn = turn
                self._save_checkpoint()
                if self.early_stopped_at_turn is not None:
                    break

        if self.early_stopped_at_turn is not None:
            for skipped_turn in range(self.early_stopped_at_turn + 1, n + 1):
                skipped_type = "converge" if skipped_turn == n else "clash"
                self.stage_reports.append(_stage2_turn_entry(
                    skipped_turn, skipped_type, [], 0.0, self.llm.model,
                    skipped=True,
                    skip_reason=(
                        f"EARLY_STOP_ENABLED=true: moderator reported converged=true at turn "
                        f"{self.early_stopped_at_turn} and the cold critic review found no blocking "
                        f"issues, so this turn never ran."
                    ),
                ))

        self._stage2_finished = True
        self._save_checkpoint()
        logger.info("── Stage 2 done in %.1fs (early stop: %s).",
                    time.monotonic() - t0,
                    f"turn {self.early_stopped_at_turn}" if self.early_stopped_at_turn else "no")

    def _paper_directory(self) -> str:
        """Label -> title (year) for every evidence paper the experts hold,
        parsed from their system prompts (the only place the runner sees them).
        The moderator writes for a reader who cannot resolve a [P#] id, so each
        write-up call carries this directory to convert citations to title+year;
        without it the moderator has been seen coining a plausible title for a
        finding an expert cited by id alone. Empty when no expert holds papers,
        and then no directory block is sent."""
        entries: Dict[str, str] = {}
        for prompt in self.experts:
            for m in re.finditer(
                    r"^\[(P?\d+)\] (.+?)\n(?:.*?\n)?  year: (\S+?)[,\n]",
                    prompt, re.M):
                entries.setdefault(m.group(1), f"{m.group(2)} ({m.group(3)})")
        if not entries:
            return ""
        lines = "\n".join(f"[{k}] {v}" for k, v in
                          sorted(entries.items(), key=lambda kv: (len(kv[0]), kv[0])))
        return ("\n\nPAPER DIRECTORY — the papers behind the researchers' [P#] "
                "citations. Use it only to name a cited paper by title and year; "
                "papers nobody cited don't belong in the write-up:\n" + lines)

    def _vantage_line(self, name: str) -> str:
        """The per-expert half of the turn-1 instruction, naming that expert's
        own line of work. The name is the first bolded run in its system prompt
        (`**Name.** Core insight …`, filled by insight.fill_system_prompt); an
        expert whose prompt has no such line gets the paper-based wording
        (STAGE2_TURN1_VANTAGE_PAPERS) rather than a broken sentence."""
        match = re.search(r"\*\*(.+?)\.?\*\*", self._prompt_of(name))
        if not match:
            # No named line of work: fall back to the paper-based wording.
            return STAGE2_TURN1_VANTAGE_PAPERS
        return _fill(STAGE2_TURN1_VANTAGE, INSIGHT_NAME=match.group(1).strip())

    def _stage1_problem_text(self) -> str:
        """The problem statement Stage 1 finalised — its last prose moderator
        record (the tally record is JSON and is skipped). Needed because a blind
        opening turn cannot read it out of the transcript."""
        for r in reversed(self.transcript):
            if (r.stage == "stage1" and r.speaker == "MODERATOR" and r.include_in_context
                    and not r.content.lstrip().startswith("{")):
                return r.content.strip()
        return self.topic

    def _propose_turn(self) -> Optional[Dict]:
        intro = STAGE2_TURN1_INTRO
        if BLIND_OPENING_TURNS:
            # "settled above" refers to the transcript, which a blind expert is
            # not shown, so the problem travels with the instruction instead.
            intro = intro.replace(
                "The problem is settled above.",
                f"The problem the group settled on:\n\n{self._stage1_problem_text()}\n",
            )
        self._add(TurnRecord(
            speaker="MODERATOR", stage="stage2", turn=1, content=intro,
            include_in_context=False,
        ))

        for name in self._rotation():
            instruction = f"{self._vantage_line(name)}\n\n{SPEECH_RULES}"
            self._add(_run_expert_speech(
                name, self._prompt_of(name), f"{intro}\n{instruction}",
                self.transcript, "stage2", 1, self.llm,
                cost_tag="stage2_propose.expert_speech",
                blind=BLIND_OPENING_TURNS,
            ))

        # moderator assembles the initial draft
        draft_req = _fill(DRAFT_REQUEST, schema=self.schema)
        draft_req += self._paper_directory()
        synth = _run_moderator_call(draft_req, self.transcript,
                                    "stage2", 1, self.llm, max_tokens=6000,
                                    cost_tag="stage2_propose.moderator")
        self._add(synth)
        draft = _extract_json(synth.content)
        if not _looks_like_proposal(draft):
            logger.error("Failed to parse initial draft JSON; keeping raw text.")
            draft = {"_raw": synth.content}
        else:
            logger.info("Initial draft assembled: \"%s\"", draft.get("title", "?"))
        self.draft_versions.append({"turn": 1, "draft": draft})
        return draft

    def _clash_turn(
        self,
        turn: int,
        draft: Optional[Dict],
        open_attacks: List[Dict],
        critic_issues: List[Dict],
    ) -> tuple[Optional[Dict], Optional[Dict]]:
        intro = _fill(
            STAGE2_CLASH_INTRO,
            turn=str(turn),
            draft=json.dumps(draft, ensure_ascii=False, indent=2),
            critic_block=self._critic_block(critic_issues),
        )
        self._add(TurnRecord(
            speaker="MODERATOR", stage="stage2", turn=turn, content=intro,
            include_in_context=False,
        ))

        for name in self._speaking_order(open_attacks, cost_tag="stage2_clash.order"):
            instruction = f"{self._attack_block(name, open_attacks)}\n\n{SPEECH_RULES}"
            self._add(_run_expert_speech(
                name, self._prompt_of(name), f"{intro}\n\n{instruction}",
                self.transcript, "stage2", turn, self.llm,
                cost_tag="stage2_clash.expert_speech",
            ))

        # moderator: turn summary + updated draft
        summary_req = _fill(SUMMARY_REQUEST, schema=self.schema)
        summary_req += self._paper_directory()
        synth = _run_moderator_call(summary_req, self.transcript,
                                    "stage2", turn, self.llm, max_tokens=8000,
                                    cost_tag="stage2_clash.moderator")
        self._add(synth)

        parsed = _extract_json(synth.content)
        summary: Optional[Dict] = None
        if isinstance(parsed, dict):
            summary = parsed.get("summary")
            updated = parsed.get("updated_draft")
            if _looks_like_proposal(updated):
                draft = updated
            elif isinstance(updated, dict):
                logger.error("Stage 2 turn %d updated_draft is not a proposal "
                             "(keys: %s); keeping the previous draft.",
                             turn, sorted(updated))
        if summary is None:
            logger.error("Failed to parse turn summary at stage2 turn %d.", turn)
        else:
            self.turn_summaries.append({"turn": turn, "summary": summary})
            logger.info(
                "Turn %d summary: %d draft update(s), %d open attack(s), %d dropped, "
                "%d settled, converged=%s.",
                turn,
                len(summary.get("draft_updates", []) or []),
                len(summary.get("open_attacks", []) or []),
                len(summary.get("dropped", []) or []),
                len(summary.get("settled", []) or []),
                summary.get("converged"),
            )
        self.draft_versions.append({"turn": turn, "draft": draft})
        return draft, summary

    def _converge_turn(self, turn: int, draft: Optional[Dict],
                       open_attacks: List[Dict], critic_issues: List[Dict]) -> None:
        intro = _fill(
            STAGE2_CONVERGE_INTRO,
            turn=str(turn),
            draft=json.dumps(draft, ensure_ascii=False, indent=2),
            critic_block=self._critic_block(critic_issues),
        )
        self._add(TurnRecord(
            speaker="MODERATOR", stage="stage2", turn=turn, content=intro,
            include_in_context=False,
        ))

        for name in self._speaking_order(open_attacks, cost_tag="stage2_converge.order"):
            instruction = f"{self._attack_block(name, open_attacks)}\n\n{SPEECH_RULES}"
            self._add(_run_expert_speech(
                name, self._prompt_of(name), f"{intro}\n\n{instruction}",
                self.transcript, "stage2", turn, self.llm,
                cost_tag="stage2_converge.expert_speech",
            ))

    # ── Stage 3 — final rewrite ───────────────────────────────────────────────

    def _stage3(self) -> tuple[List[Dict], str]:
        call_snap = self.llm.call_log_snapshot()
        t0 = time.monotonic()
        logger.info("── Stage 3 (final rewrite) started.")
        request = _fill(FINAL_REWRITE_REQUEST, schema=self.schema)
        request += self._paper_directory()
        synth = _run_moderator_call(request, self.transcript,
                                    "stage3", 1, self.llm, max_tokens=FINAL_PROPOSAL_MAX_TOKENS,
                                    cost_tag="stage3.moderator")
        raw = synth.content
        self._add(synth)

        ideas: List[Dict] = []
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            ideas = [parsed]
        elif isinstance(parsed, list):
            ideas = [x for x in parsed if isinstance(x, dict)]
            if len(ideas) > 1:
                logger.warning("Stage 3 returned %d ideas, truncating to 1.", len(ideas))
                ideas = ideas[:1]
        if ideas and not _looks_like_proposal(ideas[0]):
            logger.error("Stage 3 JSON parsed but is not a proposal (keys: %s); "
                         "discarding.", sorted(ideas[0]))
            ideas = []

        duration = time.monotonic() - t0
        if not ideas:
            logger.error("Failed to parse Stage 3 proposal JSON. Raw: %s", raw[:500])
        else:
            logger.info("── Stage 3 done in %.1fs — final proposal: \"%s\"",
                        duration, ideas[0].get("title", "?"))
        self.stage_reports.append(cost_report.make_stage_entry(
            stage_id="debate.stage3",
            label="Debate Stage 3 — Final rewrite (single structured proposal)",
            calls=self.llm.calls_since(call_snap), duration_sec=duration, model=self.llm.model,
        ))
        return ideas, raw

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def run(self) -> DebateResult:
        t0 = time.monotonic()
        usage0 = self.llm.usage_snapshot()
        resumed = self._stage1_turns_done > 0 or self._stage2_finished
        logger.info("=== Debate %s: %s (%d experts, m=%d, n=%d) ===",
                    "resumed" if resumed else "started",
                    self.topic, self.k, self._m, self._n)

        if self._stage1_turns_done < self._m:
            self._stage1()
        if not self._stage2_finished:
            self._stage2()
        ideas, raw_synthesis = self._stage3()

        usage = self.llm.usage_since(usage0)
        logger.info(
            "=== Debate complete in %.1fs. %d ideas. API: %d calls, %d in + %d out tokens ≈ %s ===",
            time.monotonic() - t0, len(ideas), usage["calls"],
            usage["input_tokens"], usage["output_tokens"],
            self.llm.format_cost(self.llm.estimated_cost_usd(usage)),
        )
        return DebateResult(
            topic=self.topic,
            transcript=self.transcript,
            final_ideas=ideas,
            raw_synthesis=raw_synthesis,
            draft_versions=self.draft_versions,
            turn_summaries=self.turn_summaries,
            critic_reports=self.critic_reports,
            early_stopped_at_turn=self.early_stopped_at_turn,
            stage_reports=self.stage_reports,
            problem_candidates=self._candidates or [],
            endorsement_votes=self.endorsement_votes,
            endorsement_tally=self.endorsement_tally,
        )

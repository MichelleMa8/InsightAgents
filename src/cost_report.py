"""
Per-topic cost and usage reporting for cost_summary.json. Each pipeline step
brackets its LLM calls with `call_log_snapshot()` / `calls_since()` and records
them via `make_stage_entry`; the entries are rolled up into per-topic totals and
a cross-topic summary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_TOKEN_KEYS = ("input", "output", "cache_write", "cache_read")

# Anthropic/OpenAI usage field name -> our token bucket key.
_USAGE_FIELD_MAP = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_creation_input_tokens": "cache_write",
    "cache_read_input_tokens": "cache_read",
}

MAJOR_STAGES = ("retrieval", "trajectory", "insight", "debate")


def _zero_tokens() -> Dict[str, int]:
    return {k: 0 for k in _TOKEN_KEYS}


def _call_type_of(cost_tag: str) -> str:
    """'stage2_clash.expert_speech' -> 'expert_speech'; 'trajectory' -> 'trajectory'."""
    return cost_tag.split(".", 1)[1] if "." in cost_tag else cost_tag


def _major_stage_of(stage_id: str) -> str:
    """'debate.stage2.turn1_propose' -> 'debate'; 'trajectory' -> 'trajectory'."""
    return stage_id.split(".", 1)[0]


def _zero_call_type_bucket() -> Dict[str, Any]:
    return {"api_calls": 0, "tokens": _zero_tokens(), "cost_usd": 0.0}


def make_stage_entry(
    stage_id: str,
    label: str,
    calls: List[Dict[str, Any]],
    duration_sec: float,
    *,
    model: Optional[str] = None,
    skipped: bool = False,
    skip_reason: Optional[str] = None,
    converged: Optional[bool] = None,
    expected_call_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate one pipeline step's raw `llm_client.call_log` slice into a
    stage entry. `expected_call_types` forces zero-value rows for roles that
    structurally belong to this stage but made no call this run (e.g. the
    speaking-order pick when ORDER_MODE="rule"), so every topic's stage
    entries have the same call_types shape for easy comparison.

    `stage_id` is the sole identifier — major stage (`_major_stage_of`) and,
    for stage2 turns, turn number/type (`_parse_debate_stage2_id`) are both
    derivable from it, so they aren't stored redundantly on the entry. Stage
    order is likewise just the entry's position in the `stages` list.
    """
    call_types: Dict[str, Dict[str, Any]] = {
        ct: _zero_call_type_bucket() for ct in (expected_call_types or [])
    }
    total_tokens = _zero_tokens()
    total_cost = 0.0
    cost_known = True

    for call in calls:
        bucket = call_types.setdefault(_call_type_of(call["stage"]), _zero_call_type_bucket())
        bucket["api_calls"] += 1
        for usage_field, token_key in _USAGE_FIELD_MAP.items():
            amount = call.get(usage_field, 0) or 0
            bucket["tokens"][token_key] += amount
            total_tokens[token_key] += amount
        if call.get("cost_usd") is None:
            cost_known = False
        else:
            bucket["cost_usd"] += call["cost_usd"]
            total_cost += call["cost_usd"]

    for bucket in call_types.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 6) if cost_known else None

    entry: Dict[str, Any] = {
        "stage_id": stage_id,
        "label": label,
        "model": model,
        "duration_sec": round(duration_sec, 1),
        "api_calls": sum(b["api_calls"] for b in call_types.values()),
        "tokens": total_tokens,
        "cost_usd": round(total_cost, 6) if cost_known else None,
        "call_types": call_types,
    }
    if skipped:
        entry["skipped"] = True
        if skip_reason:
            entry["skip_reason"] = skip_reason
    if converged is not None:
        entry["converged"] = converged
    return entry


def build_topic_totals(stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sum a topic's stages into overall totals plus a `by_major_stage` rollup.
    A major_stage key is present in the rollup only if at least one stage
    entry with that major_stage exists — its absence means the topic never
    reached that step (e.g. it errored out during retrieval), which is
    different from reaching it and it costing $0.
    """
    total_calls = 0
    total_tokens = _zero_tokens()
    total_cost = 0.0
    cost_known = True
    total_duration = 0.0
    by_major: Dict[str, Dict[str, Any]] = {}

    for stage in stages:
        total_calls += stage["api_calls"]
        for k in _TOKEN_KEYS:
            total_tokens[k] += stage["tokens"][k]
        total_duration += stage["duration_sec"]
        if stage["cost_usd"] is None:
            cost_known = False
        else:
            total_cost += stage["cost_usd"]

        bucket = by_major.setdefault(_major_stage_of(stage["stage_id"]), {
            "api_calls": 0, "cost_usd": 0.0, "duration_sec": 0.0, "_cost_known": True,
        })
        bucket["api_calls"] += stage["api_calls"]
        bucket["duration_sec"] += stage["duration_sec"]
        if stage["cost_usd"] is None:
            bucket["_cost_known"] = False
        else:
            bucket["cost_usd"] += stage["cost_usd"]

    topic_cost = round(total_cost, 6) if cost_known else None
    for stage_name, bucket in by_major.items():
        known = bucket.pop("_cost_known")
        bucket["cost_usd"] = round(bucket["cost_usd"], 6) if known else None
        bucket["duration_sec"] = round(bucket["duration_sec"], 1)
        pct = (bucket["cost_usd"] / topic_cost * 100) if (known and topic_cost) else 0.0
        bucket["pct_of_topic_cost"] = round(pct, 2)

    return {
        "api_calls": total_calls,
        "tokens": total_tokens,
        "cost_usd": topic_cost,
        "duration_sec": round(total_duration, 1),
        "by_major_stage": by_major,
    }


def _parse_debate_turn_id(prefix: str, stage_id: str) -> Optional[tuple]:
    """'debate.stage2.turn2_clash' with prefix 'debate.stage2.turn' -> (2, 'clash')."""
    if not stage_id.startswith(prefix):
        return None
    rest = stage_id[len(prefix):]
    num_str, _, turn_type = rest.partition("_")
    try:
        return int(num_str), turn_type
    except ValueError:
        return None


def _parse_debate_stage1_id(stage_id: str) -> Optional[tuple]:
    """'debate.stage1.turn2_endorsement' -> (2, 'endorsement'). None if not a stage1 turn."""
    return _parse_debate_turn_id("debate.stage1.turn", stage_id)


def _parse_debate_stage2_id(stage_id: str) -> Optional[tuple]:
    """'debate.stage2.turn2_clash' -> (2, 'clash'). None if not a stage2 turn."""
    return _parse_debate_turn_id("debate.stage2.turn", stage_id)


def build_cross_topic_summary(per_topic: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cross-topic sums + averages, computed straight from each topic's
    `stages`/`totals` (already-built per_topic entries), not re-derived from
    raw call logs. Averages for a given major_stage / debate turn only run
    over topics that actually reached that step and didn't skip it — a
    skipped early-stop turn (0 calls, 0 cost) would otherwise silently pull
    the average down and misrepresent what that turn typically costs.
    """
    topics = list(per_topic.values())
    num_topics = len(topics)
    num_completed = sum(1 for t in topics if t["status"] == "completed")
    num_failed = sum(1 for t in topics if t["status"] == "failed")
    num_early_stopped = sum(1 for t in topics if t.get("early_stopped_at_turn") is not None)

    sum_calls = sum(t["totals"]["api_calls"] for t in topics)
    sum_duration = sum(t["totals"]["duration_sec"] for t in topics)
    cost_known = all(t["totals"]["cost_usd"] is not None for t in topics)
    sum_cost = sum(t["totals"]["cost_usd"] for t in topics if t["totals"]["cost_usd"] is not None)

    avg_calls = round(sum_calls / num_topics, 2) if num_topics else 0.0
    avg_cost = round(sum_cost / num_topics, 6) if (num_topics and cost_known) else None
    avg_duration = round(sum_duration / num_topics, 2) if num_topics else 0.0

    # ── average_by_major_stage: only over topics that reached that stage ──
    average_by_major_stage: Dict[str, Any] = {}
    for major in MAJOR_STAGES:
        reached = [t["totals"]["by_major_stage"][major] for t in topics if major in t["totals"]["by_major_stage"]]
        n = len(reached)
        if n == 0:
            average_by_major_stage[major] = {
                "n_topics_ran": 0, "avg_api_calls": 0.0, "avg_cost_usd": None, "avg_duration_sec": 0.0,
                "pct_of_avg_total_cost": 0.0,
            }
            continue
        m_calls = sum(b["api_calls"] for b in reached) / n
        m_duration = sum(b["duration_sec"] for b in reached) / n
        m_known = all(b["cost_usd"] is not None for b in reached)
        m_cost = (sum(b["cost_usd"] for b in reached) / n) if m_known else None
        pct = (m_cost / avg_cost * 100) if (m_cost is not None and avg_cost) else 0.0
        average_by_major_stage[major] = {
            "n_topics_ran": n,
            "avg_api_calls": round(m_calls, 2),
            "avg_cost_usd": round(m_cost, 6) if m_cost is not None else None,
            "avg_duration_sec": round(m_duration, 2),
            "pct_of_avg_total_cost": round(pct, 2),
        }

    # ── average_by_debate_substage: stage1 (by turn) / stage2 (by turn) / stage3 ─
    stage3_entries = [s for t in topics for s in t["stages"] if s["stage_id"] == "debate.stage3"]

    def _avg_of(entries: List[Dict[str, Any]], *, exclude_skipped: bool = False) -> Dict[str, Any]:
        use = [e for e in entries if not (exclude_skipped and e.get("skipped"))]
        n = len(use)
        if n == 0:
            return {"n_topics_ran": 0, "avg_api_calls": 0.0, "avg_cost_usd": None, "avg_duration_sec": 0.0}
        m_calls = sum(e["api_calls"] for e in use) / n
        m_duration = sum(e["duration_sec"] for e in use) / n
        known = all(e["cost_usd"] is not None for e in use)
        m_cost = (sum(e["cost_usd"] for e in use) / n) if known else None
        return {
            "n_topics_ran": n,
            "avg_api_calls": round(m_calls, 2),
            "avg_cost_usd": round(m_cost, 6) if m_cost is not None else None,
            "avg_duration_sec": round(m_duration, 2),
        }

    def _debate_substage_summary(parse_fn) -> Dict[str, Any]:
        """
        Average a multi-turn debate substage (stage1 or stage2): a `by_turn`
        breakdown (averaged directly over each turn's entries), plus a
        whole-substage average computed by summing each topic's own turns
        into one per-topic total FIRST, then averaging across topics —
        pooling individual turn-instances instead would average per-turn,
        not per-topic, and misrepresent "average substage cost per topic".
        """
        by_turn: Dict[int, List[Dict[str, Any]]] = {}
        for t in topics:
            for stage in t["stages"]:
                parsed = parse_fn(stage["stage_id"])
                if parsed:
                    turn_num, _turn_type = parsed
                    by_turn.setdefault(turn_num, []).append(stage)

        by_turn_summary: Dict[str, Any] = {}
        for turn_num in sorted(by_turn):
            entries = by_turn[turn_num]
            _, turn_type = parse_fn(entries[0]["stage_id"])
            key = f"turn{turn_num}_{turn_type}"
            summary = _avg_of(entries, exclude_skipped=True)
            if len(entries) > summary["n_topics_ran"]:
                summary["note"] = (
                    f"Averaged only over the {summary['n_topics_ran']}/{len(entries)} topic(s) that actually "
                    f"ran this turn; the rest early-stopped before it — not directly comparable to turns "
                    f"that ran in every topic."
                )
            by_turn_summary[key] = summary

        topic_totals: List[Dict[str, Any]] = []
        for t in topics:
            turn_entries = [s for s in t["stages"] if parse_fn(s["stage_id"])]
            if not turn_entries:
                continue
            known = all(e["cost_usd"] is not None for e in turn_entries)
            topic_totals.append({
                "api_calls": sum(e["api_calls"] for e in turn_entries),
                "duration_sec": sum(e["duration_sec"] for e in turn_entries),
                "cost_usd": sum(e["cost_usd"] for e in turn_entries) if known else None,
            })
        summary = _avg_of(topic_totals)
        summary["by_turn"] = by_turn_summary
        return summary

    average_by_debate_substage = {
        "stage1": _debate_substage_summary(_parse_debate_stage1_id),
        "stage2": _debate_substage_summary(_parse_debate_stage2_id),
        "stage3": _avg_of(stage3_entries),
    }

    return {
        "num_topics": num_topics,
        "num_completed": num_completed,
        "num_failed": num_failed,
        "early_stop_rate": round(num_early_stopped / num_topics, 4) if num_topics else 0.0,
        "sum": {
            "api_calls": sum_calls,
            "cost_usd": round(sum_cost, 6) if cost_known else None,
            "duration_sec": round(sum_duration, 2),
        },
        "average_per_topic": {
            "api_calls": avg_calls,
            "cost_usd": avg_cost,
            "duration_sec": avg_duration,
        },
        "average_by_major_stage": average_by_major_stage,
        "average_by_debate_substage": average_by_debate_substage,
    }

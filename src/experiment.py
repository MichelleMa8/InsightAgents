"""
Main experiment runner: topic in, research proposal out.

For each topic:
  1-3. Build the expert system prompts (config.EXPERTISE_SOURCE selects how):
         semantic_scholar (online)  seed paper -> Semantic Scholar recommendation
                                    graph -> insights, one per expert
         contribution     (offline) seed paper -> local citation graph -> the
                                    same insight call
  4.   Run the multi-agent debate (debate/runner.py)
  5.   Save outputs/<timestamp>/<topic_slug>/result.json

Examples (run from src/):
  python experiment.py                                     # all topics in data/topics.json
  python experiment.py "GUI agent" "Graph RAG"             # specific topics
  python experiment.py --topics-file ../data/topics.json   # topics from another file
  python experiment.py --seed-papers-file ../data/seed_papers_example.json
                                                           # seed papers instead of topics (online only)
  python experiment.py --expertise-source offline          # override config.EXPERTISE_SOURCE
  python experiment.py --dry-run [topics]                  # steps 1-3 only
  python experiment.py --jobs 4                            # 4 topics concurrently
  python experiment.py --resume 20260626-170921 [topics]   # continue an interrupted run

Options:
  topics                  topics to run (default: all from the topics file)
  --topics-file PATH      JSON topics file (default: data/topics.json)
  --seed-papers-file PATH JSON seed-paper file; semantic_scholar source only
  --expertise-source SRC  semantic_scholar | contribution (aliases: online | offline)
  --dry-run               stop after insight generation
  --jobs, -j N            topics run concurrently in threads (default 1)
  --resume TIMESTAMP      resume outputs/<TIMESTAMP>/ from each topic's checkpoints

Outputs, per run directory outputs/<timestamp>/:
  <topic_slug>/progress.json   checkpoint after every completed step and debate turn
  <topic_slug>/result.json     written once the topic completes; marks it done
  cost_summary.json            LLM cost and usage per topic and stage
  run_<timestamp>.log          console log (a resumed run starts a new log file)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

# Allow running from src/ as working directory
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
import cost_report
from debate.runner import DebateRunner
from expertise import ALIASES, SOURCES, get_backend, resolve_source
from llm_client import LLMClient
from topics import (
    DEFAULT_TOPICS_FILE,
    TopicSpec,
    load_seed_paper_specs,
    load_topic_specs,
)

# The topic the calling thread is working on, so concurrent topics stay
# readable in the interleaved log. Every LogRecord carries it as `topic`
# (empty in the main thread), including records from library loggers.
_TOPIC_CTX = threading.local()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(topic)s %(message)s"


def _record_factory_with_topic(_base=logging.getLogRecordFactory()):
    def factory(*args, **kwargs):
        record = _base(*args, **kwargs)
        topic = getattr(_TOPIC_CTX, "name", "")
        record.topic = f" [{topic}]" if topic else ""
        return record
    return factory


logging.setLogRecordFactory(_record_factory_with_topic())

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()],
)
for _noisy in ("httpx", "httpcore", "openai", "anthropic"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _add_run_log_file(run_dir: Path) -> Path:
    """
    Attach a FileHandler for this invocation's log alongside the existing
    console handler, so every run is saved to <run_dir>/run_<timestamp>.log
    with no need to manually pipe through `tee`. The timestamp is stamped at
    log-file-creation time (not the run directory's timestamp), so resuming
    an existing run_dir later creates a new, separate log file rather than
    overwriting the original. FileHandler flushes on every emitted record
    (inherited from StreamHandler), so the file updates in real time.
    """
    log_path = run_dir / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    return log_path


# ── Progress file (per-topic checkpoint state) ────────────────────────────────

PROGRESS_SCHEMA_VERSION = "1.0"


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON via a temp file + os.replace so a crash mid-write can never
    leave a truncated file — progress.json must survive exactly the scenario
    it exists for."""
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _new_progress(topic: Optional[str], expertise_source: str = "contribution") -> dict:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "topic": topic,
        # Which backend produced this checkpoint's Steps 1-3. The two backends
        # write different payloads into the same slots, so a resume under the
        # other source is refused rather than silently misread.
        "expertise_source": expertise_source,
        "pretrain_started_at": datetime.now().isoformat(),
        "pretrain_completed_at": None,
        "pretrain_stages": [],
        "retrieval": None,
        "trajectory": None,
        "insight": None,
        "debate_started_at": None,
        "debate": None,
        "error": None,
    }


def _record_error(progress: dict, failed_step: Optional[str], message: str) -> None:
    """failed_step names a pretrain step whose outcome (not just its
    execution) failed, so a resume must redo it; None marks an interruption
    whose checkpoints are all still valid."""
    progress["error"] = {
        "failed_step": failed_step,
        "message": message,
        "at": datetime.now().isoformat(),
    }


def _reset_failed_step(progress: dict) -> None:
    """Clear a recorded error before retrying. A step-level failure ("no
    seed paper found", "no valid paper graph") invalidates that step's
    checkpoint, so its state block and stage entry are dropped and the step
    reruns; an interruption (crash, API error) leaves every checkpoint valid,
    so nothing is reset and the run simply continues."""
    error = progress.get("error")
    if not error:
        return
    failed_step = error.get("failed_step") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    if failed_step in ("retrieval", "trajectory", "insight"):
        progress[failed_step] = None
        progress["pretrain_stages"] = [
            s for s in progress["pretrain_stages"] if s["stage_id"] != failed_step
        ]
        logger.info("Retrying failed step '%s' (previous error: %s).", failed_step, message)
    else:
        logger.info("Resuming after interruption (previous error: %s).", message)
    progress["error"] = None


def _load_progress(topic: Optional[str], out_dir: Path,
                   expertise_source: str = "contribution") -> dict:
    progress_file = out_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, encoding="utf-8") as f:
            progress = json.load(f)
        saved_source = progress.get("expertise_source")
        if saved_source != expertise_source:
            raise RuntimeError(
                f"{progress_file} was built with EXPERTISE_SOURCE='{saved_source}', but this "
                f"run uses '{expertise_source}'. Resume it under its own source, or start a "
                f"new run directory."
            )
        _reset_failed_step(progress)
        return progress
    return _new_progress(topic, expertise_source)


# ── Per-topic pipeline ────────────────────────────────────────────────────────

def run_topic_debate(
    progress: dict,
    llm_client: LLMClient,
    save: Callable[[], None],
) -> dict:
    """
    Step 4: run (or resume, from progress["debate"]) the debate — a checkpoint
    is saved after every turn — then assemble the full result dict.

    The debate half of result.json is identical whatever built the expertise;
    the Steps 1-3 half comes from the backend that produced the checkpoints
    (progress["expertise_source"]), so a result file always describes the
    pipeline that actually ran.
    """
    topic = progress["topic"]
    logger.info("[4/4] Running debate for '%s' …", topic)

    if progress["debate_started_at"] is None:
        progress["debate_started_at"] = datetime.now().isoformat()

    def on_checkpoint(state: dict) -> None:
        progress["debate"] = state
        save()

    backend = get_backend(progress["expertise_source"])
    runner = DebateRunner(
        topic, progress["insight"]["expert_prompts"], llm_client,
        checkpoint=progress["debate"], on_checkpoint=on_checkpoint,
    )
    debate = runner.run()
    debate_completed_at = datetime.now().isoformat()

    return {
        "topic": topic,
        **backend.result_section(progress),
        "debate_transcript": [
            {
                "speaker": r.speaker,
                "stage": r.stage,
                "turn": r.turn,
                "content": r.content,
            }
            for r in debate.transcript
        ],
        "problem_candidates": debate.problem_candidates,
        "endorsement_votes": debate.endorsement_votes,
        "endorsement_tally": debate.endorsement_tally,
        "draft_versions": debate.draft_versions,
        "turn_summaries": debate.turn_summaries,
        "critic_reports": debate.critic_reports,
        "early_stopped_at_turn": debate.early_stopped_at_turn,
        "debate_started_at": progress["debate_started_at"],
        "debate_completed_at": debate_completed_at,
        "debate_stages": debate.stage_reports,
        "final_ideas": debate.final_ideas,
    }


# ── Topic-level parallelism ───────────────────────────────────────────────────

_THREAD_CLIENTS = threading.local()


def _thread_clients(api_key: str) -> tuple[LLMClient, LLMClient]:
    """
    (debate_client, insight_client) for the calling thread, created on first
    use and reused afterwards — so a sequential run still builds exactly one
    pair, and --jobs N builds N (ThreadPoolExecutor reuses its threads).

    Threads must not share a client: cost accounting is done by taking a
    call_log_snapshot() and slicing calls_since() off the end of the log, and
    that "everything appended since" interval is only meaningful with a single
    writer. Two topics sharing a client would attribute each other's calls.
    """
    pair = getattr(_THREAD_CLIENTS, "pair", None)
    if pair is None:
        pair = _THREAD_CLIENTS.pair = (
            LLMClient(cfg.LLM_PROVIDER, cfg.DEBATE_MODEL, api_key, cfg.MAX_TOKENS_PER_SPEECH),
            LLMClient(cfg.LLM_PROVIDER, cfg.INSIGHT_MODEL, api_key, 2000),
        )
    return pair


def _preload_shared_caches(topic_hint: str) -> None:
    """
    Warm the process-wide read-only singletons before any worker thread starts.

    They are all lazy ("load on first use, keep forever") and none of them is
    guarded by a lock, so N threads hitting a cold one all load it: harmless
    for the small ones, but forward_references.json is 348 MB of JSON that
    expands to several GB in memory, and the embedding model is a multi-second
    load. Paying for each of them once, up front, keeps that out of the pool.
    """
    from graph import acl_abstracts, impact, loader
    from retrieval.searcher import embed_topic

    logger.info("Preloading shared caches before starting workers …")
    t0 = time.time()
    embed_topic(topic_hint)     # embedding model weights
    impact.preload()            # forward_references.json (~348 MB)
    loader.preload()            # paper manifest
    acl_abstracts.build_index() # ACL title -> abstract index
    from expertise import contribution_graph
    from retrieval import searcher
    contribution_graph._paper_years()   # corpus_id -> year (anachronism check)
    searcher._build_reverse_lut()       # contribution_id -> shard row (neighbour similarity)
    logger.info("Shared caches ready in %.1fs.", time.time() - t0)


def _run_one_topic(spec: TopicSpec, run_dir: Path, dry_run: bool, api_key: str,
                   expertise_source: str) -> None:
    """
    Run one unit of work end to end into run_dir/<slug>/. Self-contained: it owns
    its output directory and checkpoint file and swallows its own failures (they
    are recorded in progress.json), so topics can run concurrently and one
    topic's crash never touches another's.
    """
    _TOPIC_CTX.name = spec.display
    try:
        out_dir  = run_dir / spec.slug
        out_file = out_dir / "result.json"
        progress_file = out_dir / "progress.json"
        out_dir.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            logger.info("Skipping '%s' — result.json already exists.", spec.display)
            return

        start = time.time()
        logger.info("━━━ Topic: %s ━━━", spec.display)
        try:
            progress = _load_progress(spec.topic, out_dir, expertise_source)
        except RuntimeError as e:
            # A checkpoint built by the other expertise source. Skipping only this
            # topic keeps the rule that one topic's failure never touches another's.
            logger.error("Cannot resume '%s': %s", spec.display, e)
            return
        debate_client, insight_client = _thread_clients(api_key)

        def save() -> None:
            _atomic_write_json(progress_file, progress)

        save()  # persist the loaded/reset/fresh state before doing any work

        try:
            get_backend(expertise_source).run_pretrain(
                spec, insight_client, progress, save, _record_error,
            )
            if progress["error"]:
                return  # step-level failure, already recorded and saved
            if progress["pretrain_completed_at"] is None:
                progress["pretrain_completed_at"] = datetime.now().isoformat()
                save()
                totals = cost_report.build_topic_totals(progress["pretrain_stages"])
                logger.info(
                    "Pretrain for '%s' done in %.1fs. API: %d call(s) ≈ %s",
                    progress["topic"], totals["duration_sec"], totals["api_calls"],
                    insight_client.format_cost(totals["cost_usd"]),
                )
            if dry_run:
                logger.info("Dry-run: stopping before debate for '%s'.", progress["topic"])
                return
            result = run_topic_debate(progress, debate_client, save)
        except Exception as e:
            logger.exception("Error running topic '%s': %s", spec.display, e)
            _record_error(progress, None, str(e))
            save()
            return

        # Atomic for the same reason as progress.json: result.json's existence
        # is the completion marker, so it must never exist half-written.
        _atomic_write_json(out_file, result)
        logger.info("Topic '%s' done in %.1fs → %s", progress["topic"], time.time() - start, out_file)
    finally:
        _TOPIC_CTX.name = ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="InsightAgents experiment runner")
    parser.add_argument("topics", nargs="*", help="Topics to run (default: all from the topics file)")
    parser.add_argument(
        "--topics-file", metavar="PATH",
        help=f"JSON topics file to run (default: {DEFAULT_TOPICS_FILE})",
    )
    parser.add_argument(
        "--seed-papers-file", metavar="PATH",
        help="JSON file of seed papers to run instead of topics (semantic_scholar "
             "source only). Entries are paper ids, or objects with a 'paper' id and "
             "an optional 'topic'; a missing topic is derived from the paper.",
    )
    parser.add_argument(
        "--expertise-source", choices=SOURCES + tuple(ALIASES), metavar="SOURCE",
        help=f"How Steps 1-3 build the expertise: {' | '.join(SOURCES)} "
             f"(or the aliases online = semantic_scholar, offline = contribution) "
             f"(default: config.EXPERTISE_SOURCE = {cfg.EXPERTISE_SOURCE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stop after insight generation (steps 1-3, checkpointed in progress.json)",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, metavar="N",
        help="Run N topics concurrently (default: 1, i.e. one after another). "
             "Everything inside a topic stays sequential.",
    )
    parser.add_argument(
        "--resume", metavar="TIMESTAMP",
        help="Continue the run in outputs/<TIMESTAMP>/ from each topic's checkpoints "
             "(any stage)",
    )
    args = parser.parse_args()

    expertise_source = resolve_source(args.expertise_source)
    # Code that defaults to the configured source reads it from config, so an
    # override belongs there rather than only in this function's locals.
    cfg.EXPERTISE_SOURCE = expertise_source

    # Positional topics win over --seed-papers-file, which wins over
    # --topics-file, which wins over the default topics file.
    try:
        if args.topics:
            specs = [TopicSpec(topic=t) for t in args.topics]
        elif args.seed_papers_file:
            if expertise_source != "semantic_scholar":
                parser.error("--seed-papers-file needs --expertise-source semantic_scholar "
                             "(the contribution source has no way to start from a paper id)")
            specs = load_seed_paper_specs(args.seed_papers_file)
        else:
            specs = load_topic_specs(args.topics_file or DEFAULT_TOPICS_FILE)
    except (OSError, ValueError) as e:
        parser.error(str(e))

    dry_run = args.dry_run
    resume_ts = args.resume
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    jobs = min(args.jobs, len(specs))

    # Validate API keys
    if cfg.LLM_PROVIDER == "anthropic" and not cfg.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")
    if cfg.LLM_PROVIDER == "openai" and not cfg.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")

    api_key = cfg.ANTHROPIC_API_KEY if cfg.LLM_PROVIDER == "anthropic" else cfg.OPENAI_API_KEY

    # Determine the run directory (timestamp-stamped)
    if resume_ts:
        run_ts = resume_ts
    else:
        run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = cfg.OUTPUT_DIR / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _add_run_log_file(run_dir)
    logger.info("Output directory: %s", run_dir)
    logger.info("Logging to: %s", log_path)

    logger.info("Expertise source: %s", expertise_source)

    # Two inputs that slugify identically would share an output directory (and,
    # in parallel, race on it), so they are rejected rather than silently
    # clobbering each other.
    by_slug: Dict[str, str] = {}
    for spec in specs:
        if spec.slug in by_slug and by_slug[spec.slug] != spec.display:
            parser.error(f"'{by_slug[spec.slug]}' and '{spec.display}' map to the same "
                         f"output directory '{spec.slug}'")
        by_slug[spec.slug] = spec.display

    if jobs == 1:
        for spec in specs:
            _run_one_topic(spec, run_dir, dry_run, api_key, expertise_source)
    else:
        logger.info("Running %d topic(s), %d at a time.", len(specs), jobs)
        if expertise_source == "contribution":
            # Local-corpus singletons only — the Semantic Scholar path loads no
            # shared data, and its API pacing is already process-wide.
            _preload_shared_caches(specs[0].topic)
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="topic") as pool:
            futures = [pool.submit(_run_one_topic, spec, run_dir, dry_run, api_key,
                                   expertise_source)
                       for spec in specs]
            try:
                for f in futures:
                    f.result()  # _run_one_topic handles its own errors; re-raise anything else
            except KeyboardInterrupt:
                # Topics that have not started are dropped; the ones in flight
                # keep their per-unit checkpoints and resume with --resume.
                logger.warning("Interrupted — cancelling topics that have not started.")
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    # ── Cost summary JSON ─────────────────────────────────────────────────────
    # Rebuilt from the on-disk progress.json / result.json files rather than just this
    # process's in-memory call logs, so it stays correct across separate
    # --resume invocations that share the same run_ts. Topics interrupted
    # mid-debate contribute their checkpointed turn entries with status
    # "in_progress" instead of losing the debate cost entirely.
    per_topic: Dict[str, dict] = {}
    topic_names: list[str] = []
    for spec in specs:
        topic = spec.display
        out_dir  = run_dir / spec.slug
        progress_file = out_dir / "progress.json"
        out_file = out_dir / "result.json"

        debate_checkpoint = None
        if progress_file.exists():
            with open(progress_file, encoding="utf-8") as f:
                progress = json.load(f)
            stages = list(progress.get("pretrain_stages", []))
            # A seed-paper run only learns its topic while running, so the
            # summary is keyed on the resolved topic when there is one.
            topic = progress.get("topic") or topic
            started_at = progress.get("pretrain_started_at")
            completed_at = progress.get("pretrain_completed_at")
            error = progress.get("error")
            error_msg = error.get("message") if isinstance(error, dict) else error
            debate_checkpoint = progress.get("debate")
            pretrain_done = bool(progress.get("insight") and progress["insight"].get("done"))
        else:
            topic_names.append(topic)
            continue  # topic never ran and left no artifact

        result = None
        if out_file.exists():
            with open(out_file, encoding="utf-8") as f:
                result = json.load(f)

        early_stopped_at_turn = None
        if error_msg is not None:
            status = "failed"
        elif pretrain_done:
            status = "pretrain_only"
        else:
            status = "in_progress"

        if result is not None:
            stages += result.get("debate_stages", [])
            completed_at = result.get("debate_completed_at", completed_at)
            if "error" in result:
                status = "failed"
                error_msg = result["error"]
            else:
                status = "completed"
                early_stopped_at_turn = result.get("early_stopped_at_turn")
        elif debate_checkpoint:
            stages += debate_checkpoint.get("stage_reports", [])
            early_stopped_at_turn = debate_checkpoint.get("early_stopped_at_turn")
            if error_msg is None:
                status = "in_progress"

        topic_names.append(topic)
        per_topic[topic] = {
            "status": status,
            "error": error_msg,
            "started_at": started_at,
            "completed_at": completed_at,
            "early_stopped_at_turn": early_stopped_at_turn,
            "totals": cost_report.build_topic_totals(stages),
            "stages": stages,
        }

    cross_topic_summary = cost_report.build_cross_topic_summary(per_topic)
    cost_summary = {
        "schema_version": "1.0",
        "run_id": run_ts,
        "generated_at": datetime.now().isoformat(),
        "config_snapshot": {
            "expertise_source": expertise_source,
            "debate_model": cfg.DEBATE_MODEL,
            "insight_model": cfg.INSIGHT_MODEL,
            "num_agents": cfg.NUM_AGENTS,
            "problem_turns": cfg.PROBLEM_TURNS,
            "debate_turns": cfg.DEBATE_TURNS,
            "critic_enabled": cfg.CRITIC_ENABLED,
            "order_mode": cfg.ORDER_MODE,
            "early_stop_enabled": cfg.EARLY_STOP_ENABLED,
            "max_tokens_per_speech": cfg.MAX_TOKENS_PER_SPEECH,
        },
        "topics": topic_names,
        "cross_topic_summary": cross_topic_summary,
        "per_topic": per_topic,
    }
    cost_file = run_dir / "cost_summary.json"
    with open(cost_file, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved cost summary → %s", cost_file)

    xts = cross_topic_summary
    logger.info(
        "=== Run totals: %d topic(s) (%d completed, %d failed), %d API call(s) ≈ %s ===",
        xts["num_topics"], xts["num_completed"], xts["num_failed"], xts["sum"]["api_calls"],
        f"${xts['sum']['cost_usd']:.2f}" if xts["sum"]["cost_usd"] is not None else "n/a",
    )


if __name__ == "__main__":
    main()

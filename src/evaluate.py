"""
Pairwise LLM-judge evaluation of final research ideas.

The judge compares two ideas on the same topic on five aspects (novelty,
significance, feasibility, clarity, effectiveness), judging both presentation
orders. A sweep compares every method with every other one per topic and
summarises the result as per-aspect Elo ratings on a radar chart.

Examples (run from src/):
  # one pair of ideas
  python evaluate.py resultA.json resultB.json
  python evaluate.py ideaA.md ideaB.md --topic "Multilingual LLM"

  # round-robin sweep over a dataset directory
  python evaluate.py --dir ../eval_data --dry-run                 # list the comparisons
  python evaluate.py --dir ../eval_data --jobs 4
  python evaluate.py --dir ../eval_data --topics <topic_dir> --methods insightagents,coi
  python evaluate.py --dir ../eval_data --plot-only               # redraw the chart only

Idea files: a run's result.json (final_ideas[--idea-index], topic read from the
file) or a plain .md/.txt idea. In --dir mode the layout is one directory per
topic with one file per method; the file stem is the method key:

  <DATASET>/<topic_dir>/insightagents.json
  <DATASET>/<topic_dir>/<method>.md

Options:
  --topic TEXT        topic for plain-text ideas outside --dir
  --idea-index N      which final_ideas entry of a result.json (default 0)
  --rounds N          judgments per presentation order (default config.EVAL_ROUNDS)
  --provider, --model judge (default config.EVAL_PROVIDER / config.EVAL_MODEL)
  --out PATH          output JSON path
  --dir only:  --topics, --methods (comma-separated filters), --jobs N,
               --overwrite (redo finished pairs), --dry-run, --plot-only,
               --no-plot, --plot-out PATH

Outputs:
  single pair  outputs/evaluations/
  sweep        evaluation/<DATASET>/eval_pairwise_<judge>.json (rewritten after
               every comparison, so an interrupted sweep resumes) and
               eval_pairwise_radar_<judge>.png (+ .pdf, + .json ratings)
"""

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from threading import Lock

import config as cfg
from evaluation.evaluator import evaluate_pairwise, load_idea, make_eval_client
from evaluation.pairwise_plot import load_sweep, plot_radar, rate_methods
from llm_client import unrecoverable_api_error
from prompts.eval_prompts import EVAL_ASPECTS
from topics import load_topics

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-call "HTTP Request: POST ... 200 OK"

EVAL_ROOT = cfg.ROOT / "evaluation"    # every sweep output lands here, mirroring the dataset name
IDEA_SUFFIXES = (".md", ".txt", ".json")


def _resolve_topic(cli_topic, *file_topics):
    topics = [t for t in file_topics if t]
    if cli_topic:
        return cli_topic
    if not topics:
        sys.exit("error: no topic found in input file(s); pass --topic")
    if len(set(topics)) > 1:
        sys.exit(f"error: input files have different topics {topics}; pass --topic to override")
    return topics[0]


def _model_slug(model: str) -> str:
    for short in ("opus", "sonnet", "haiku"):
        if short in model:
            return short
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


def _write_record(record, llm, out_arg, run_ts):
    record["usage"] = llm.usage_snapshot()
    record["cost_usd"] = llm.estimated_cost_usd()
    if out_arg:
        out_path = Path(out_arg)
    else:
        out_path = cfg.OUTPUT_DIR / "evaluations" / run_ts / "eval_pairwise.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Cost: {llm.format_cost(record['cost_usd'])} "
          f"({llm.usage['calls']} calls, {llm.usage['input_tokens']} in / {llm.usage['output_tokens']} out tokens)")


# ── pairwise sweep over a dataset directory ───────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@lru_cache(maxsize=1)
def _topic_slug_index() -> dict:
    """slug -> display name, from the topics file.

    Read lazily and best-effort: a dataset whose topics are not in the topics
    file falls back to the directory name.
    """
    try:
        current = load_topics()
    except (OSError, ValueError) as e:
        logging.warning("Topic display names unavailable: %s", e)
        current = []
    by_slug = {}
    for topic in current:
        by_slug.setdefault(_slug(topic), topic)
    return by_slug


def _topic_display(dir_name: str) -> str:
    """Recover the judge-facing topic from a topic directory slug, e.g.
    'open_ended_generation_evaluation' -> 'Open-ended generation evaluation'."""
    return _topic_slug_index().get(_slug(dir_name), dir_name.replace("_", " "))


def _dataset_dir(arg: str) -> Path:
    path = Path(arg)
    if not path.is_dir():
        sys.exit(f"error: no such dataset directory: {arg}")
    return path.resolve()


def _csv_filter(raw):
    """--topics/--methods value -> set of names, or None for 'all'."""
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _sweep_out_path(dataset_dir: Path, model: str, rounds: int, out_arg=None) -> Path:
    """One JSON per (dataset, judge model, rounds), under evaluation/<dataset>/."""
    if out_arg:
        return Path(out_arg)
    suffix = "" if rounds == 1 else f"_{rounds}-round"
    return EVAL_ROOT / dataset_dir.name / f"eval_pairwise_{_model_slug(model)}{suffix}.json"


def _topic_methods(topic_dir: Path, only_methods):
    """{method key: idea file} for one topic — every idea file directly in it,
    keyed by its stem."""
    methods = {}
    for path in sorted(topic_dir.iterdir()):
        if not path.is_file() or path.suffix not in IDEA_SUFFIXES:
            continue
        if only_methods and path.stem not in only_methods:
            continue
        if path.stem in methods:
            sys.exit(f"error: two idea files for method {path.stem!r} in {topic_dir}")
        methods[path.stem] = path
    return methods


def _pair_id(topic_dir: str, method_a: str, method_b: str) -> str:
    return f"{topic_dir}|{method_a}|{method_b}"


def _collect_pairwise_jobs(dataset_dir, only_topics, only_methods, done_ids):
    """One job per (topic, unordered pair of methods). `done_ids` are the pairs
    already present in the sweep file, skipped unless --overwrite cleared it."""
    jobs, skipped = [], []
    for topic_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        topic_name = topic_dir.name
        if only_topics and topic_name not in only_topics and _slug(topic_name) not in {_slug(t) for t in only_topics}:
            continue

        methods = _topic_methods(topic_dir, only_methods)
        if len(methods) < 2:
            print(f"  [skip] {topic_name}: {len(methods)} method(s) found, need 2 to compare")
            continue

        for method_a, method_b in combinations(sorted(methods), 2):
            pair_id = _pair_id(topic_name, method_a, method_b)
            if pair_id in done_ids:
                skipped.append(pair_id)
                continue
            jobs.append({
                "pair_id": pair_id,
                "topic_dir": topic_name,
                "topic": _topic_display(topic_name),
                "method_a": method_a,
                "method_b": method_b,
                "a_path": methods[method_a],
                "b_path": methods[method_b],
            })
    return jobs, skipped


def _run_one_pairwise(job, args, run_ts):
    """Judge one pair with its own client so per-comparison usage and cost stay
    exact even when jobs run in parallel."""
    llm = make_eval_client(args.provider, args.model)
    idea_a, topic_a = load_idea(job["a_path"], args.idea_index)
    idea_b, topic_b = load_idea(job["b_path"], args.idea_index)
    # A result.json carries its own topic; plain-text ideas fall back to the
    # topic recovered from the directory name.
    topic = topic_a or topic_b or job["topic"]
    result = evaluate_pairwise(llm, idea_a, idea_b, topic, args.rounds)

    return {
        "pair_id": job["pair_id"],
        "topic": topic,
        "topic_dir": job["topic_dir"],
        "method_a": job["method_a"],
        "method_b": job["method_b"],
        "idea_paths": [_repo_rel(job["a_path"]), _repo_rel(job["b_path"])],
        "timestamp": run_ts,
        **result,
        "usage": llm.usage_snapshot(),
        "cost_usd": llm.estimated_cost_usd(),
    }


def _job_label(job) -> str:
    return f"{job['topic_dir']}: {job['method_a']} vs {job['method_b']}"


def _repo_rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(cfg.ROOT))
    except ValueError:
        return str(path)


def _load_sweep_file(path: Path):
    """Existing sweep file -> {pair_id: comparison}, or {} if there is none."""
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read existing sweep file {path} ({exc}); "
                 f"delete it or pass --out to write elsewhere")
    return {c.get("pair_id") or _pair_id(c.get("topic_dir", ""), c["method_a"], c["method_b"]): c
            for c in blob.get("comparisons", [])}


def _write_sweep_file(path: Path, comparisons, dataset_dir, args, model, provider, run_ts):
    """Rewrite the whole sweep file. Called after every comparison so an
    interrupted run keeps everything it has already paid for; written to a temp
    file first so an interrupt mid-write cannot leave a truncated JSON."""
    ordered = sorted(comparisons.values(),
                     key=lambda c: (c["topic_dir"], c["method_a"], c["method_b"]))
    record = {
        "mode": "pairwise",
        "dataset": dataset_dir.name,
        "dataset_dir": _repo_rel(dataset_dir),
        "aspects": EVAL_ASPECTS,
        "provider": provider,
        "model": model,
        "rounds": args.rounds,
        "timestamp": run_ts,
        "methods": sorted({m for c in ordered for m in (c["method_a"], c["method_b"])}),
        "topics": sorted({c["topic_dir"] for c in ordered}),
        "comparisons": ordered,
        "total_cost_usd": round(sum(c.get("cost_usd") or 0.0 for c in ordered), 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)
    return record


def _run_batch_pairwise(args, run_ts):
    dataset_dir = _dataset_dir(args.dir)
    model = args.model or cfg.EVAL_MODEL
    provider = args.provider or cfg.EVAL_PROVIDER
    out_path = _sweep_out_path(dataset_dir, model, args.rounds, args.out)

    print(f"=== Pairwise sweep: {dataset_dir} ===")
    print(f"    judge={provider}/{model}  rounds={args.rounds}  jobs={args.jobs}")
    print(f"    out={_repo_rel(out_path)}")

    comparisons = {} if args.overwrite else _load_sweep_file(out_path)
    jobs, skipped = _collect_pairwise_jobs(
        dataset_dir, _csv_filter(args.topics), _csv_filter(args.methods), set(comparisons),
    )
    for pair_id in skipped:
        print(f"  [have] {pair_id}")
    if not jobs:
        print("\nEvery comparison is already done (use --overwrite to redo them) "
              "— going straight to the summary chart.")
        if args.dry_run:
            return
        if comparisons:
            _write_sweep_file(out_path, comparisons, dataset_dir, args, model, provider, run_ts)
        _write_pairwise_chart(out_path, dataset_dir, args, model, provider)
        return

    print(f"\n{len(jobs)} comparison(s) to run:")
    for job in jobs:
        print(f"  {job['topic_dir']:<55} {job['method_a']:<15} vs {job['method_b']}")
    if args.dry_run:
        return
    print()

    # An account-level failure (bad key, no model access, exhausted quota) would
    # hit every remaining job identically, so the first one stops the sweep
    # instead of replaying the same error a dozen times. A list, not a flag, so
    # it carries the reason; append/truth-test are atomic enough for --jobs > 1
    # (a lost race only means two workers both report the same abort).
    aborted: list[str] = []
    write_lock = Lock()

    def _guarded(job):
        if aborted:
            return job, None, None      # never started
        try:
            comparison = _run_one_pairwise(job, args, run_ts)
        except Exception as exc:        # one bad pair must not sink the sweep
            fatal = unrecoverable_api_error(exc)
            if fatal and not aborted:
                aborted.append(fatal)
            print(f"  [FAIL] {_job_label(job)}: {exc}", flush=True)
            return job, None, exc
        with write_lock:                # one writer at a time over the shared file
            comparisons[job["pair_id"]] = comparison
            _write_sweep_file(out_path, comparisons, dataset_dir, args, model, provider, run_ts)
        wins = "  ".join(f"{a}={comparison['aggregate'][a]['winner']}" for a in EVAL_ASPECTS)
        print(f"  [done] {_job_label(job)}: {wins}", flush=True)
        return job, comparison, None

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(pool.map(_guarded, jobs))
    else:
        outcomes = [_guarded(job) for job in jobs]

    done = sum(1 for _, c, _ in outcomes if c is not None)
    run_cost = sum(c["cost_usd"] or 0.0 for _, c, _ in outcomes if c is not None)
    failures = [job for job, _, exc in outcomes if exc is not None]
    not_started = [job for job, c, exc in outcomes if c is None and exc is None]

    if aborted:
        print(f"\n[abort] {aborted[0]}")
        print(f"        This failure applies to every call, so the sweep stopped: "
              f"{len(not_started)} comparison(s) not attempted. Fix the account/model "
              f"issue and rerun — finished comparisons are skipped automatically.")

    print(f"\n{done}/{len(outcomes)} comparison(s) written to {_repo_rel(out_path)}, "
          f"judge cost ${run_cost:.4f}")
    _write_pairwise_chart(out_path, dataset_dir, args, model, provider)
    if failures or not_started:
        sys.exit(f"{len(failures)} comparison(s) failed"
                 + (f", {len(not_started)} not attempted" if not_started else ""))


def _radar_path(sweep_path: Path) -> Path:
    """Chart beside the sweep it summarizes: eval_pairwise_opus.json ->
    eval_pairwise_radar_opus.png. A --out with some other name just gets
    '_radar' appended, so the ratings JSON can never land on the sweep file."""
    stem = sweep_path.stem
    stem = (stem.replace("eval_pairwise", "eval_pairwise_radar", 1)
            if "eval_pairwise" in stem else stem + "_radar")
    return sweep_path.with_name(stem + ".png")


def _write_pairwise_chart(sweep_path, dataset_dir, args, model, provider):
    """Elo radar over every comparison in the sweep file, saved beside it."""
    if args.no_plot:
        return
    records = load_sweep(sweep_path)
    if not records:
        print(f"\n[plot] no comparisons in {_repo_rel(sweep_path)} — no chart written")
        return

    summary = rate_methods(records)
    summary.update({"dataset": dataset_dir.name, "model": model,
                    "provider": provider, "rounds": args.rounds})

    png_path = Path(args.plot_out) if args.plot_out else _radar_path(sweep_path)
    json_path = png_path.with_suffix(".json")
    if json_path.resolve() == sweep_path.resolve():
        sys.exit(f"error: the chart's ratings file would overwrite the sweep file "
                 f"{_repo_rel(sweep_path)}; pass a different --plot-out")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Elo over {summary['records']} comparison(s), "
          f"{len(summary['topics'])} topic(s) ===")
    width = max(len(m) for m in summary["methods"])
    for method in sorted(summary["methods"], key=lambda m: -summary["ratings"][m]["average"]):
        row = summary["ratings"][method]
        aspects = "  ".join(f"{a[:4]}={row[a]:.0f}" for a in summary["aspects"])
        print(f"  {method:<{width}}  avg={row['average']:.0f}   {aspects}")

    try:
        title = f"{dataset_dir.name} — judge {model}"
        plot_radar(summary, png_path, title=title)
    except ImportError as exc:
        print(f"[plot] matplotlib is missing ({exc}); wrote ratings only: {_repo_rel(json_path)}")
        return
    print(f"Saved chart:   {_repo_rel(png_path)}  (+ .pdf)")
    print(f"Saved ratings: {_repo_rel(json_path)}")


def _run_plot_only(args):
    dataset_dir = _dataset_dir(args.dir)
    model = args.model or cfg.EVAL_MODEL
    provider = args.provider or cfg.EVAL_PROVIDER
    sweep_path = _sweep_out_path(dataset_dir, model, args.rounds, args.out)
    print(f"=== Pairwise chart: {_repo_rel(sweep_path)} ===")
    if not sweep_path.is_file():
        sys.exit(f"error: no sweep file at {sweep_path} — run the sweep first")
    _write_pairwise_chart(sweep_path, dataset_dir, args, model, provider)


def main():
    p = argparse.ArgumentParser(description="Pairwise LLM-judge evaluation of research ideas")
    p.add_argument("ideas", nargs="*", help="2 idea files to compare: result.json or .md/.txt")
    p.add_argument("--dir", help="sweep mode: every method against every other in a dataset "
                                 "directory (<dir>/<topic_dir>/<method>.{md,txt,json})")
    p.add_argument("--topics", help="--dir only: comma-separated topic dirs to include")
    p.add_argument("--methods", help="--dir only: comma-separated method keys (file stems) to include")
    p.add_argument("--jobs", type=int, default=1, help="--dir only: comparisons to run in parallel")
    p.add_argument("--overwrite", action="store_true", help="--dir only: redo comparisons already in the sweep file")
    p.add_argument("--dry-run", action="store_true", help="--dir only: list the jobs and exit")
    p.add_argument("--plot-only", action="store_true",
                   help="--dir only: skip judging, just redraw the chart from the sweep file")
    p.add_argument("--no-plot", action="store_true", help="--dir only: skip the summary chart")
    p.add_argument("--plot-out", default=None,
                   help="--dir only: chart path (default: alongside the sweep file)")
    p.add_argument("--topic", help="research topic (required for plain-text ideas outside --dir)")
    p.add_argument("--idea-index", type=int, default=0, help="final_ideas index in result.json inputs (default 0)")
    p.add_argument("--rounds", type=int, default=cfg.EVAL_ROUNDS,
                   help="independent judgments per presentation order")
    p.add_argument("--provider", choices=["anthropic", "openai"], default=None,
                   help=f"default: config.EVAL_PROVIDER ({cfg.EVAL_PROVIDER})")
    p.add_argument("--model", default=None, help=f"default: config.EVAL_MODEL ({cfg.EVAL_MODEL})")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: --dir -> evaluation/<dataset>/eval_pairwise_<model>.json, "
                        "single pair -> outputs/evaluations/)")
    args = p.parse_args()
    run_ts = f"{datetime.now():%Y%m%d-%H%M%S}"

    if args.dir:
        if args.ideas:
            sys.exit("error: pass either --dir or idea files, not both")
        if args.plot_only:
            _run_plot_only(args)
        else:
            _run_batch_pairwise(args, run_ts)
        return

    if len(args.ideas) != 2:
        sys.exit(f"error: expected 2 idea files, got {len(args.ideas)}")

    llm = make_eval_client(args.provider, args.model)
    loaded = [load_idea(path, args.idea_index) for path in args.ideas]
    topic = _resolve_topic(args.topic, *(t for _, t in loaded))

    record = {
        "mode": "pairwise",
        "topic": topic,
        "aspects": EVAL_ASPECTS,
        "idea_paths": args.ideas,
        "provider": llm.provider,
        "model": llm.model,
        "rounds": args.rounds,
        "timestamp": run_ts,
    }
    result = evaluate_pairwise(llm, loaded[0][0], loaded[1][0], topic, args.rounds)
    record.update(result)
    print(f"\n=== Pairwise: A={args.ideas[0]}  B={args.ideas[1]} ===")
    for aspect, agg in result["aggregate"].items():
        print(f"  {aspect:<13} winner={agg['winner']:<4} (A:{agg['A']} B:{agg['B']} tie:{agg['tie']})")
    _write_record(record, llm, args.out, run_ts)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\ninterrupted")
    except Exception as exc:
        # A bad key / no model access / exhausted quota is a setup problem, not a
        # bug: report it as one line instead of a stack trace. Everything else
        # keeps its traceback.
        reason = unrecoverable_api_error(exc)
        if not reason:
            raise
        sys.exit(f"\nerror: {reason}")

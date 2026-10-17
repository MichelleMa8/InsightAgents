# Usage

All commands run from `src/`.

## Command-line options

```bash
python experiment.py                                   # all topics in data/topics.json
python experiment.py "Graph retrieval augmented generation"   # specific topic(s)
python experiment.py --topics-file ../data/topics.json
python experiment.py --seed-papers-file ../data/seed_papers_example.json   # online only
python experiment.py --expertise-source online|offline # default: config.EXPERTISE_SOURCE
python experiment.py --dry-run                         # stop after insight generation
python experiment.py --jobs 4                          # run 4 topics concurrently
python experiment.py --resume <TIMESTAMP>              # continue an interrupted run
```

When several inputs are given, the first available one is used, in this order:
topics on the command line → `--seed-papers-file` → `--topics-file` →
`data/topics.json`.

## Input files

**Topics** (`data/topics.json`) work in both modes. For each topic, the pipeline
searches for candidate seed papers and the LLM picks one. If the graph grown from
that seed fails validation, the next candidate is tried.

```json
[{"id": "T001", "topic": "Assessing moral values encoded in large language models"}]
```

Plain strings (`["Topic A", "Topic B"]`) work too.

**Seed papers** (`--seed-papers-file`) work in online mode only. Each entry fixes
the paper the graph is grown from, and the seed search is skipped. There is no
fallback: if that seed's graph fails validation, the entry fails.

```json
[{"id": "S001", "topic": "Graph retrieval augmented generation",
  "paper": "6586f17b316a65df944e84ddbaf0e3aeab029a75"}]
```

- `paper` is any Semantic Scholar paper id: the 40-hex `paperId` from a
  `semanticscholar.org/paper/…` URL, `ARXIV:…`, `DOI:…`, `CorpusId:…` or `URL:…`.
  These are not SCG corpus ids.
- `topic` is optional. If it is missing, the LLM derives one from the paper's
  title and abstract.

## Outputs

Each run writes to `outputs/<TIMESTAMP>/`:

| file | content |
| --- | --- |
| `<topic>/result.json` | final idea (`final_ideas`), paper graph, insights with evidence, full debate record |
| `<topic>/progress.json` | per-stage checkpoint used by `--resume` |
| `run_<ts>.log` | console log |
| `cost_summary.json` | token / cost accounting |

## Configuration

All parameters live in `src/config.py`:

- **Models:** `LLM_PROVIDER`, `DEBATE_MODEL`, `INSIGHT_MODEL`, `EVAL_MODEL`
- **Expertise source:** `EXPERTISE_SOURCE`
- **Graph size:** `S2_KEEP_PER_LEVEL` (online), `CG_KEEP_PER_LEVEL` (offline)
- **Debate:** `NUM_AGENTS`, `PROBLEM_TURNS`, `DEBATE_TURNS`, `CRITIC_ENABLED`,
  `EARLY_STOP_ENABLED`
- **Semantic Scholar:** set `S2_API_KEY` in `.env` to get a reserved
  1 request/second. Responses are cached in `data/s2_cache/`.

# Pairwise Evaluation

An LLM judge compares two ideas on the same topic. It picks a winner (or a tie)
for each of five aspects: novelty, significance, feasibility, clarity and
effectiveness. Both presentation orders are judged to cancel position bias. The protocol
(aspects, order swapping, Elo summary) follows the Idea Arena of
[CoI-Agent](https://arxiv.org/abs/2410.13185) ([code](https://github.com/DAMO-NLP-SG/CoI-Agent)). All
commands run from `src/`. The judge model is set by `EVAL_MODEL` in
`src/config.py`, or `--model`.

## Compare two ideas

An idea is a `result.json` or a `.md` / `.txt` file:

```bash
python evaluate.py ../outputs/<RUN>/<topic>/result.json other_idea.md
python evaluate.py ideaA.md ideaB.md --topic "Graph retrieval augmented generation"
```

## Round-robin sweep

Create one directory per topic, with one idea file per method in it. The file
stem is the method name:

```
eval_data/
└── <topic_dir>/
    ├── insightagents.json      # ours: a copied outputs/<RUN>/<topic>/result.json
    ├── coi.md                  # other systems, as markdown / plain text
    └── ai_scientist.md
```

```bash
python evaluate.py --dir ../eval_data --dry-run      # list the comparisons
python evaluate.py --dir ../eval_data --jobs 4       # judge every method pair per topic
python evaluate.py --dir ../eval_data --methods insightagents,coi --topics <topic_dir>
python evaluate.py --dir ../eval_data --plot-only    # redraw the chart only
```

The topic shown to the judge is taken from the first available source:

1. the `topic` field in a `result.json`
2. a matching entry in `data/topics.json`
3. the directory name

Finished pairs are skipped when you rerun the sweep; pass `--overwrite` to redo
them. Results:

```
evaluation/<dataset>/eval_pairwise_<judge>.json        every comparison, with judge rationales
evaluation/<dataset>/eval_pairwise_radar_<judge>.png   per-aspect Elo radar (+ .pdf, + .json)
```

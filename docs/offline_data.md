# Offline Data: Scientific Contribution Graph

The offline mode (`--expertise-source offline`) runs on the
**Scientific Contribution Graph (SCG)**
([Jansen, 2026](https://arxiv.org/abs/2605.15011) ·
[GitHub](https://github.com/cognitiveailab/scientific-contribution-graph)).
SCG covers ACL Anthology papers. For each paper it stores the extracted
contributions and the precursor links between them. It also ships
forward-reference indices and precomputed `Qwen/Qwen3-Embedding-0.6B`
embeddings.

## Download

We use **release 1.0**, which is hosted in a Hugging Face storage bucket:

```bash
pip install -U "huggingface_hub[hf_transfer]"

mkdir -p data/sci-cont-graph && cd data/sci-cont-graph
hf buckets ls   hf://buckets/pajansen/scientific-contribution-graph/releases-tar/1.0/
hf buckets sync hf://buckets/pajansen/scientific-contribution-graph/releases-tar/1.0 . --verbose
sha256sum -c scg-release-1.0.0.tar.gz.sha256
tar -xzf scg-release-1.0.0.tar.gz        # creates ./data/
```

If you already have a copy elsewhere, a symlink works too:
`ln -s /path/to/scg-release data/sci-cont-graph`.

## Expected layout

No conversion is needed. The code reads the extracted release directly from
`data/sci-cont-graph/data/` (`DATA_DIR` in `src/config.py`):

```
data/sci-cont-graph/data/
├── papers/<first 2 digits of corpus_id>/<corpus_id>.json   # one paper + its contributions
├── metadata/
│   ├── forward_references.json        # contribution -> what builds on it
│   ├── paper_manifest.jsonl
│   ├── corpus_id_to_paper_year.json
│   └── paper_title_to_corpus_id.json
└── embeddings/
    ├── manifest.json                  # Qwen3-Embedding-0.6B, dim 1024
    ├── shard_*.npy
    └── shard_*.lut.json               # row -> contribution_id
```

## Built automatically on first run

- **Query embedder.** `Qwen/Qwen3-Embedding-0.6B` is fetched from the Hugging
  Face Hub and runs on CPU.
- **ACL abstracts.** SCG papers carry no abstracts. On first use, the ACL
  Anthology bulk BibTeX export (~90 MB) is downloaded and indexed into
  `data/acl_cache/`.

The offline mode needs a few GB of RAM.

> Newer SCG releases (e.g. 1.1 under `releases-tar/current/`) use the same layout.
> Our results were produced on 1.0, and `CORPUS_MAX_YEAR` in `src/config.py` is
> set for that release.

<div align="center">
  <img src="assets/logo.svg" alt="InsightAgents" width="96" />
  <h1>InsightAgents</h1>
  <p><strong>Research idea generation through a debate among insight-grounded expert agents</strong></p>
  <p>
    <!-- TODO: replace "#" with the arXiv link once the paper is out -->
    <a href="#"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" /></a>
    <a href="#quick-start"><img alt="Quick Start" src="https://img.shields.io/badge/Quick-Start-2563eb?style=for-the-badge" /></a>
    <a href="docs/evaluation.md"><img alt="Evaluation" src="https://img.shields.io/badge/Pairwise-Evaluation-7c3aed?style=for-the-badge" /></a>
  </p>
</div>

**InsightAgents** turns a research topic into a research proposal. It first grows
a graph of related papers from a seed paper. From that graph it distills several
*insights* (each one is how a line of work sees its field), and gives each insight
to its own expert agent. The experts then debate under a moderator and a critic,
and the debate ends in one structured proposal.

The paper graph can be built in two ways:

| Mode | Paper source | Setup |
| --- | --- | --- |
| **Online** (default) | live [Semantic Scholar API](https://www.semanticscholar.org/product/api) | an API key is recommended |
| **Offline** | local [Scientific Contribution Graph](https://github.com/cognitiveailab/scientific-contribution-graph) | [download the data](docs/offline_data.md) |

## Installation

```bash
git clone https://github.com/<org>/InsightAgents.git && cd InsightAgents
conda create -n insightagents python=3.12 -y && conda activate insightagents
pip install -r requirements.txt
cp .env.example .env    # set OPENAI_API_KEY / ANTHROPIC_API_KEY, and optionally S2_API_KEY
```

## Quick Start

```bash
cd src

# online: build the paper graph from the Semantic Scholar API
python experiment.py "Graph retrieval augmented generation"

# offline: build it from the local Scientific Contribution Graph (see docs/offline_data.md)
python experiment.py "Graph retrieval augmented generation" --expertise-source offline

# a batch of topics, or of seed papers (online only)
python experiment.py --topics-file ../data/topics.json --jobs 4
python experiment.py --seed-papers-file ../data/seed_papers_example.json
```

Each topic writes its final idea and the full debate record to
`outputs/<TIMESTAMP>/<topic>/result.json`. Runs are checkpointed, so an
interrupted run can be continued with `--resume <TIMESTAMP>`.

**Pairwise evaluation.** The LLM judge compares ideas from different systems,
following the Idea Arena protocol of [CoI-Agent](https://arxiv.org/abs/2410.13185):

```bash
python evaluate.py --dir ../eval_data --jobs 4
```

## Documentation

- [Usage](docs/usage.md): CLI options, input file formats, outputs, configuration
- [Offline data](docs/offline_data.md): download and set up the Scientific Contribution Graph
- [Pairwise evaluation](docs/evaluation.md): dataset layout, judging, Elo radar

## Citation

```bibtex
@article{insightagents2026,
  title   = {InsightAgents},
  author  = {TBD},
  journal = {arXiv preprint},
  year    = {2026}
}
```

The offline mode builds on the Scientific Contribution Graph, and the pairwise
evaluation follows CoI-Agent:

```bibtex
@article{jansen2026scientific,
  title   = {The Scientific Contribution Graph: Automated Literature-based Technological Roadmapping at Scale},
  author  = {Jansen, Peter A.},
  journal = {arXiv preprint arXiv:2605.15011},
  year    = {2026}
}
```

```bibtex
@article{li2024chain,
  title   = {Chain of Ideas: Revolutionizing Research Via Novel Idea Development with LLM Agents},
  author  = {Li, Long and Xu, Weiwen and Guo, Jiayan and Zhao, Ruochen and Li, Xingxuan and Yuan, Yuqian and Zhang, Boqiang and Jiang, Yuming and Xin, Yifei and Dang, Ronghao and Zhao, Deli and Rong, Yu and Feng, Tian and Bing, Lidong},
  journal = {arXiv preprint arXiv:2410.13185},
  year    = {2024}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
for attribution and acknowledgment of government support.

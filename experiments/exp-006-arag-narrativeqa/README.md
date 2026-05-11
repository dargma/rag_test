# exp-006: A-RAG on NarrativeQA dev_10_doc

Wrapper around [Ayanami0730/arag](https://github.com/Ayanami0730/arag) (paper
[arXiv:2602.03442](https://arxiv.org/abs/2602.03442)) for the 293-query
NarrativeQA benchmark used by the other three RAG systems in this repo.

## What this directory contains

| File | Purpose |
|---|---|
| `run.py`            | End-to-end driver: build data → build index → batch agent → eval |
| `build_data.py`     | Convert `reproduce/dataset/narrativeqa_dev_10_doc*.json` → A-RAG's `chunks.json` / `questions.json` |
| `eval.py`           | Score `predictions.jsonl` with the shared `eval_metrics.py` (max-over-2-refs F1/EM/ROUGE-L/BLEU/METEOR), raw + cleaned outputs |
| `clean_answer.py`   | Strip A-RAG's `(Chunk ID: …)` and "*supported by…*" tail (post-hoc concise-answer normalization) |
| `configs/narrativeqa.yaml` | A-RAG agent config: GPT-4.1-mini, SBERT embedder, 15-loop max |
| `results/`          | Generated outputs (`predictions.jsonl`, `eval_summary.json`) |

## Prerequisites

- A-RAG repo cloned as a sibling of `rag_test/` (default), or pointed to by `ARAG_PATH` env var or `config/paths.yaml`. See repo-root `EXTERNAL_REPOS.md` for clone + checkout commands.
- A-RAG installed (`pip install -e .` inside the arag repo). The repo-root `scripts/setup_env.sh` does this.
- `OPENAI_API_KEY` (or `ARAG_API_KEY`) exported.

## Usage

```bash
# from rag_test/ root
export OPENAI_API_KEY="sk-..."
python experiments/exp-006-arag-narrativeqa/run.py
```

Optional flags:

- `--skip_build_data`   — reuse existing chunks/questions
- `--skip_build_index`  — reuse existing sentence_index.pkl
- `--workers N`         — concurrent agent workers (default 5)
- `--limit N`           — smoke test on first N queries

Wall-clock: ~12 min with 5 workers on a single CPU+API (no GPU). Cost: ~$2 on GPT-4.1-mini for full 293.

## Expected results (fair-matched run, n=293)

|        | raw | cleaned |
|---     |---: |---:     |
| F1     | 6.61 | **12.07** |
| ROUGE-L| 5.84 | 10.44 |
| METEOR | 18.94 | **24.31** |
| EM     | 0.00 | 0.00 |

The "cleaned" column post-strips chunk citations / explanation tails (analogous
to ComoRAG's `### Final Answer` extraction). See `reports/RAG_COMPARISON_NARRATIVEQA.md`
§3.1 (caveat 3) and §4.8 for context.

## Why two columns?

A-RAG's paper-default system prompt asks the agent to *"Cite the specific chunks
that support your answer"*. Raw outputs therefore average **85 tokens** with
chunk-ID citations — vs the baselines' 3–20 tokens. Token-level F1 is unfair to
verbose answers, so we report both. See §4.8 of the comparison report.

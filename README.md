# rag_test — Fair-Compare 4-Way RAG Orchestrator

A reproducible, side-by-side comparison of four retrieval-augmented generation
systems on the NarrativeQA `dev_10_doc` benchmark
(293 queries × 4 111 passages, 140-token chunks):

| System | F1 (gpt-4.1-mini) | Repo |
|---|---|---|
| **HippoRAG 2** | **24.72** | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| **ComoRAG** | 23.09 | [EternityJune25/ComoRAG](https://github.com/EternityJune25/ComoRAG) |
| **RAPTOR** | 15.64 | [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) |
| **A-RAG** raw / cleaned | 6.61 / 12.07 | [Ayanami0730/arag](https://github.com/Ayanami0730/arag) |

Full system-by-system analysis: [`reports/RAG_COMPARISON_NARRATIVEQA.md`](reports/RAG_COMPARISON_NARRATIVEQA.md).
Step-by-step reproduction: [`REPRODUCE.md`](REPRODUCE.md).

---

## What's in this repo

`rag_test/` is **not** a fork of any single RAG system. It's an
**orchestrator** that drives four upstream RAG repos as sibling directories
and produces a fair, identically-configured comparison. The four upstream
repos are kept pristine; any minimal patches needed (only ComoRAG today) live
under `patches/` and are applied by `scripts/setup_env.sh`.

```
rag_test/
├── REPRODUCE.md            ← step-by-step reproduction guide (start here)
├── EXTERNAL_REPOS.md       ← frozen sibling commits + patch info
├── scripts/                ← setup_env.sh, run_all.sh, build_comparison.py, verify_results.py
├── experiments/            ← per-system run wrappers + shared eval_metrics.py
│   ├── exp-003-hipporag2-gpt/
│   ├── exp-004-raptor-gpt/
│   ├── exp-005-comorag-narrativeqa/
│   └── exp-006-arag-narrativeqa/
├── patches/comorag/        ← 2 patches applied to upstream ComoRAG
├── utils/paths.py          ← env > yaml > sibling path resolver
├── config/                 ← paths.yaml.example
├── reproduce/dataset/      ← NarrativeQA + HotpotQA + MuSiQue + 2WikiMHQA dev splits (bundled)
└── reports/                ← RAG_COMPARISON_NARRATIVEQA.md + METHODOLOGY/PROGRESS/TRACKER
```

The upstream RAG systems themselves are **not vendored** — they live as
siblings of this repo, cloned by you via `git clone` and pinned in
`EXTERNAL_REPOS.md`.

---

## Quickstart

```bash
mkdir -p ~/rag-eval && cd ~/rag-eval

# 1) clone the four upstream RAG repos + this orchestrator (siblings)
for r in OSU-NLP-Group/HippoRAG parthsarthi03/raptor \
         EternityJune25/ComoRAG Ayanami0730/arag    \
         dargma/rag_test; do
  git clone "https://github.com/$r"
done

# 2) install everything (checkout frozen hashes, apply patches, pip install -e)
cd rag_test
bash scripts/setup_env.sh

# 3) OpenAI key
export OPENAI_API_KEY="sk-..."

# 4) run all four systems
bash scripts/run_all.sh

# 5) verify the numbers match the reference (±0.5 F1)
python scripts/verify_results.py --tolerance 0.5
```

Detailed walkthrough (prerequisites, expected outcomes, troubleshooting,
customisation, file map) lives in [`REPRODUCE.md`](REPRODUCE.md).

---

## Why a separate orchestrator?

The four upstream repos use different chunkers, embedders, prompts, and
evaluation scripts. Comparing their published numbers across-the-board is
unsafe. `rag_test/` enforces:

1. **Identical input split**: HippoRAG 2's `narrativeqa_dev_10_doc.json`
   (293 queries × 4 111 × 140-token passages) — bundled in `reproduce/dataset/`.
2. **Identical LLM backbone**: `gpt-4.1-mini` via OpenAI API in all four wrappers.
3. **Identical retriever embedder**: `sentence-transformers/multi-qa-mpnet-base-cos-v1`.
4. **Identical evaluator**: a single `experiments/eval_metrics.py` (token-F1,
   EM, ROUGE-L, BLEU-1/4, METEOR) — applied to every system's outputs.
5. **Identical hardware budget**: ~14–56 min / system on one A100-80GB; CPU
   also works.
6. **Reproducibility verifier**: `scripts/verify_results.py` exits 0 only if
   every metric on every system is within ±0.5 of the reference values
   recorded in `REPRODUCE.md` §2.

---

## License

MIT — orchestration code, wrappers, scripts, and patches in this repo.
Upstream RAG systems retain their own licenses (cloned separately, see
`EXTERNAL_REPOS.md`).

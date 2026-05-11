# Reproduce — RAG Comparison on NarrativeQA `dev_10_doc`

A step-by-step guide to reproduce the 4-way RAG comparison
(HippoRAG 2 vs RAPTOR vs ComoRAG vs A-RAG) on the
[HippoRAG 2 official benchmark split](https://huggingface.co/datasets/osunlp/HippoRAG_2)
**`narrativeqa_dev_10_doc`** (293 queries × 4 111 passages × 140 tokens each).

The reference table is at [`reports/RAG_COMPARISON_NARRATIVEQA.md`](reports/RAG_COMPARISON_NARRATIVEQA.md).

---

## TL;DR (≤ 10 commands)

```bash
mkdir -p ~/rag-eval && cd ~/rag-eval

# 1) clone five repos as siblings
for r in OSU-NLP-Group/HippoRAG parthsarthi03/raptor \
         EternityJune25/ComoRAG Ayanami0730/arag    \
         dargma/rag_test; do
  git clone "https://github.com/$r"
done

# 2) install everything (checkout frozen hashes, apply patches, pip install -e)
cd rag_test
bash scripts/setup_env.sh

# 3) provide OpenAI credentials
export OPENAI_API_KEY="sk-..."

# 4) run all four systems + auto-build comparison
bash scripts/run_all.sh

# 5) verify the numbers match the canonical reference (±0.5 F1)
python scripts/verify_results.py --tolerance 0.5
```

Wall-clock: ~2.5 h end-to-end on one A100-80GB (or any modern CPU + OpenAI API).
Cost: ~$5–8 of GPT-4.1-mini API.

The final table lands at `reports/RAG_COMPARISON_NARRATIVEQA_table.md` and
`verify_results.py` exits 0 only when every metric is within tolerance of the
reference values in §2 below.

---

## 1. Prerequisites

| Tool / dep | Why |
|---|---|
| **Python 3.10+** | All four wrappers and the eval scorer |
| **git** | Cloning external repos |
| **OpenAI account + API key** | All four systems use `gpt-4.1-mini` as the LLM backbone |
| **GPU (optional)** | Speeds up sentence-embedding for the larger systems. CPU works for everything. |

### 1.1 Pre-flight self-check (30 s)

Run these four commands before §2 to confirm your environment is ready:

```bash
python3 --version                     # expect 3.10+
git --version                         # any modern version
df -h .                               # expect ≥ 4 GB free on the working volume
echo "${OPENAI_API_KEY:0:7}..."       # expect prefix like 'sk-proj' or 'sk-svc'
```

If any line fails the expectation, fix that single dependency before continuing.

**No local LLM weights are needed** — the GPT-4.1-mini backbone is called over
the OpenAI API. The only model that downloads to disk is the SBERT embedder
`sentence-transformers/multi-qa-mpnet-base-cos-v1` (~440 MB), pulled
automatically by `sentence-transformers` into `~/.cache/huggingface/hub/` on
first use.

### Disk space breakdown

| Component | Size |
|---|---|
| `rag_test/` (this repo) — incl. **`reproduce/dataset/narrativeqa_dev_10_doc*.json` (~95 MB)** | ~100 MB |
| 4 external repos (HippoRAG + raptor + ComoRAG + arag) after `pip install -e .` | ~1.5 GB |
| SBERT model cache (`~/.cache/huggingface`) | ~440 MB |
| Generated indices (KG / tree / memory-pool / sentence-index, cached after first run) | ~1.5 GB |
| **Total** | **~3.5 GB** |

The wrappers are not GPU-locked. Embedding precomputation runs in seconds on
GPU and a few minutes on CPU.

---

## 2. Expected outcomes (reference for verification)

A successful reproduction lands within ±0.5 F1 of these reference values on
NarrativeQA `dev_10_doc` (293 queries, gpt-4.1-mini, 140-token chunks).
`scripts/verify_results.py` (§7.1) compares your results file-by-file against
this table.

| System | F1 | EM | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | Wall-clock (1× A100) |
|---|---|---|---|---|---|---|---|
| **HippoRAG 2** (`exp-003`) | **24.72** | 6.14 | 23.54 | 19.85 | 1.30 | 31.06 | 56 min |
| **RAPTOR** (`exp-004`) | 15.64 | 0.00 | 14.16 | 11.00 | 0.62 | 27.81 | 46 min |
| **ComoRAG** (`exp-005`) | **23.09** | 8.53 | 22.37 | 19.77 | 0.53 | 24.39 | 23 min |
| **A-RAG raw** (`exp-006`) | 6.61 | 0.00 | 5.84 | 3.71 | 0.17 | 18.94 | 14 min |
| **A-RAG cleaned** (`exp-006`) | 12.07 | 0.00 | 10.44 | 7.84 | 0.41 | 24.31 | — (post-processing) |

Sources (each test asserts the metrics in this file):
- `experiments/exp-003-hipporag2-gpt/results/hipporag2_gpt_metrics.json`
- `experiments/exp-004-raptor-gpt/results/raptor_gpt_metrics.json`
- `experiments/exp-005-comorag-narrativeqa/results/comorag_metrics.json`
- `experiments/exp-006-arag-narrativeqa/results/eval_summary.json` (`.raw` / `.cleaned`)

**Why A-RAG raw is so low**: the published A-RAG prompt asks the agent to cite
chunk IDs in its answer (`(Chunk ID: 42)`). The "cleaned" column post-strips
these tails. See `reports/RAG_COMPARISON_NARRATIVEQA.md` §4.8.

**Acceptable variation**: ±0.5 F1 is the default tolerance in
`verify_results.py`. Larger deviations usually mean (a) the OpenAI model
changed sampling defaults, (b) a sibling repo is on a different commit than
documented in `EXTERNAL_REPOS.md`, or (c) the SBERT embedder cache is stale.

---

## 3. Clone the five repositories

The four external systems must live as siblings of `rag_test/`. Default layout:

```
<parent>/
├── HippoRAG/
├── raptor/
├── ComoRAG/
├── arag/
└── rag_test/      ← this repo
```

```bash
mkdir -p ~/rag-eval && cd ~/rag-eval
git clone https://github.com/OSU-NLP-Group/HippoRAG.git
git clone https://github.com/parthsarthi03/raptor.git
git clone https://github.com/EternityJune25/ComoRAG.git
git clone https://github.com/Ayanami0730/arag.git
git clone https://github.com/dargma/rag_test.git
```

Cloning elsewhere is fine — see `EXTERNAL_REPOS.md` for env-var / yaml overrides.

**The NarrativeQA `dev_10_doc` dataset is bundled with this repo** —
`reproduce/dataset/narrativeqa_dev_10_doc.json` (~90 MB) and
`narrativeqa_dev_10_doc_corpus.json` (~2.9 MB) are checked into git, so
`git clone rag_test` brings the data with it. No separate dataset download is
required. (HotpotQA / MuSiQue / 2WikiMultiHopQA splits are also bundled for
future ablations.)

---

## 4. Install dependencies + apply patches

From the `rag_test/` directory:

```bash
cd rag_test
bash scripts/setup_env.sh
```

This script:
1. Installs common Python deps (`sentence-transformers`, `pyyaml`, `tiktoken`, `openai`, …).
2. Checks out the frozen commit hashes documented in `EXTERNAL_REPOS.md` for ComoRAG and A-RAG.
3. Applies the three ComoRAG patches in `patches/comorag/` (lazy-import vLLM + SBERT routing).
4. `pip install -e .` for each of the four sibling repos.

Verify the path resolution:

```bash
python utils/paths.py
# expect:  hipporag   → /.../HippoRAG  [OK]
#          raptor     → /.../raptor    [OK]
#          comorag    → /.../ComoRAG   [OK]
#          arag       → /.../arag      [OK]
```

If one is `[MISSING]`, either set the corresponding env var (`HIPPORAG_PATH`,
`RAPTOR_PATH`, …) or copy `config/paths.yaml.example` → `config/paths.yaml`
and edit.

---

## 5. Provide the OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

All four wrappers read this single variable. The A-RAG wrapper additionally
re-exports it as `ARAG_API_KEY` internally.

Optional:
```bash
export ARAG_BASE_URL="https://api.openai.com/v1"   # default
export ARAG_MODEL="gpt-4.1-mini"                    # default
```

---

## 6. Run all four systems

### 6.1 All four at once

```bash
bash scripts/run_all.sh
```

Each experiment writes its own `results/` directory:

| Experiment | Wall-clock (single A100) | Key output |
|---|---|---|
| `exp-003-hipporag2-gpt`     | ~56 min (indexing 35 min + QA 21 min) | `results/hipporag2_gpt_metrics.json` |
| `exp-004-raptor-gpt`        | ~46 min (indexing 39 min + QA  7 min) | `results/raptor_gpt_metrics.json` |
| `exp-005-comorag-narrativeqa` | ~23 min (indexing 14 min + QA  9 min) | `results/comorag_metrics.json` |
| `exp-006-arag-narrativeqa`  | ~14 min (indexing  2 min + QA 12 min) | `results/eval_summary.json` |

### 6.2 One method at a time (standalone)

Each `run.py` is **self-contained** — no orchestrator dependency. Pick any of:

```bash
# HippoRAG 2
python experiments/exp-003-hipporag2-gpt/run.py

# RAPTOR
python experiments/exp-004-raptor-gpt/run.py

# ComoRAG (140-tok)
python experiments/exp-005-comorag-narrativeqa/run.py

# ComoRAG (512-tok, native chunking)  — optional ablation
python experiments/exp-005-comorag-narrativeqa/run_512.py

# A-RAG
python experiments/exp-006-arag-narrativeqa/run.py
```

### 6.3 A subset via env var

```bash
WHICH="hipporag2 raptor" bash scripts/run_all.sh   # any subset of: hipporag2 raptor comorag arag
```

---

## 7. Aggregate the comparison table and verify

### 7.1 Build the auto-table

`scripts/run_all.sh` calls this automatically at the end, but you can rerun
it any time:

```bash
python scripts/build_comparison.py
```

Output:
- **stdout**: a Markdown table with F1 / EM / ROUGE-L / BLEU-1 / BLEU-4 / METEOR.
- **`reports/RAG_COMPARISON_NARRATIVEQA_table.md`**: the same table, persisted.

The hand-curated narrative comparison in
**`reports/RAG_COMPARISON_NARRATIVEQA.md`** (system descriptions, fairness
analysis, latency, recommendations) is independent of this auto-table.

### 7.2 Verify against the canonical reference

```bash
python scripts/verify_results.py --tolerance 0.5
```

`verify_results.py` reads each system's metrics JSON and compares every
metric against the reference table in §2 of this document. Default tolerance
is ±0.5 (absolute). Output looks like:

```
system         |              F1 |              EM | ...
hipporag2      |  24.72 (+0.00)✓ |   6.14 (+0.00)✓ | ...
raptor         |  15.64 (+0.00)✓ |   0.00 (+0.00)✓ | ...
comorag        |  23.09 (+0.00)✓ |   8.53 (+0.00)✓ | ...
arag_raw       |   6.61 (-0.00)✓ |   0.00 (+0.00)✓ | ...
arag_cleaned   |  12.07 (+0.00)✓ |   0.00 (+0.00)✓ | ...
✓ All systems within tolerance of reference values.
```

Exit code is `0` if every metric is within tolerance, `1` otherwise (suitable
for CI gating). Use `--which hipporag2 raptor` to verify a subset, or
`--tolerance 1.0` if you ran a different commit of HippoRAG / RAPTOR and need
more slack.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src.hipporag'` | HippoRAG sibling missing or not pip-installed — re-run `bash scripts/setup_env.sh`. |
| `ImportError: VLLMOfflineOpenIE` (ComoRAG) | ComoRAG patches not applied — run `git apply patches/comorag/*.patch` in the ComoRAG repo. |
| `openai.AuthenticationError` | `OPENAI_API_KEY` not exported. |
| `openai.BadRequestError: Unrecognized request argument: reasoning_effort` | Editing `experiments/exp-006-.../configs/narrativeqa.yaml`? Comment out `reasoning_effort` — it is only valid for `o1` / `gpt-5` models, not `gpt-4.1-mini`. |
| A-RAG runs but gets very low F1 (≤ 7) | Expected for the **raw** score. See §2 of this document and `reports/RAG_COMPARISON_NARRATIVEQA.md` §4.8 — A-RAG cites chunk IDs in its answers (paper-published prompt); the **cleaned** column in `eval_summary.json` post-strips them. |
| `verify_results.py` reports `✗` on a metric | Check (a) `EXTERNAL_REPOS.md` commit hashes match your siblings, (b) the OpenAI model name in each wrapper is still `gpt-4.1-mini` (OpenAI sometimes ships sampling-default changes), (c) the SBERT embedder cache is not stale (`rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--multi-qa-mpnet-base-cos-v1` and rerun). |
| `KeyError: 'id'` in A-RAG `build_index.py` | The `build_data.py` step was skipped. Re-run `experiments/exp-006-arag-narrativeqa/build_data.py`. |

---

## 9. Customising the run

| Want to … | Do … |
|---|---|
| Change LLM backbone | Edit each wrapper's `LLM_NAME` / `model` setting. Token-level metrics in `eval_metrics.py` are LLM-independent. |
| Use a different embedder | Same — edit the wrapper. Note ComoRAG's `embedding_model/__init__.py` routing if the new name doesn't match `mpnet` / `minilm` / `multi-qa` / `all-…` (see patch #2). |
| Run a different NarrativeQA split | Drop `<other_split>.json` + `<other_split>_corpus.json` into `reproduce/dataset/` and update `DATASET_NAME` in each wrapper. |
| Add a 5th system | Create `experiments/exp-007-<system>/run.py`, register it in `scripts/run_all.sh` and `scripts/build_comparison.py`. |

---

## 10. Where things live (file map)

```
rag_test/
├── REPRODUCE.md                ← this file
├── EXTERNAL_REPOS.md           ← frozen commit hashes + patch info
├── config/
│   └── paths.yaml.example      ← copy to paths.yaml to override sibling defaults
├── utils/
│   └── paths.py                ← env > yaml > sibling resolution
├── scripts/
│   ├── setup_env.sh            ← one-shot install
│   ├── run_all.sh              ← run all 4 systems + build comparison
│   ├── build_comparison.py     ← aggregate 4 metric JSONs → md table
│   └── verify_results.py       ← compare each system's metrics against §2 reference
├── patches/comorag/            ← 3 patches required for ComoRAG
├── experiments/
│   ├── eval_metrics.py         ← shared token-F1 / EM / ROUGE-L / BLEU / METEOR
│   ├── exp-003-hipporag2-gpt/  ← canonical fair-compare runs (gpt-4.1-mini)
│   ├── exp-004-raptor-gpt/
│   ├── exp-005-comorag-narrativeqa/
│   ├── exp-006-arag-narrativeqa/   ← A-RAG wrapper (build_data, run, eval, clean_answer)
│   ├── exp-001-hipporag2-narrativeqa/   ← DEPRECATED — local-LLM variant
│   └── exp-002-raptor-narrativeqa/      ← DEPRECATED — local-LLM variant
├── reproduce/
│   └── dataset/                ← narrativeqa_dev_10_doc{,_corpus}.json (tracked, ~93 MB)
└── reports/
    ├── RAG_COMPARISON_NARRATIVEQA.md       ← hand-curated narrative comparison
    └── RAG_COMPARISON_NARRATIVEQA_table.md ← auto-generated by build_comparison.py
```

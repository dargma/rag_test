# External Repos — Frozen Versions for Reproducibility

`rag_test/` is an **orchestrator**: it does not vendor any of the four RAG
systems being compared. Instead, it expects them to live as **siblings** of
this repository (default), and provides wrappers under `experiments/exp-00X/`
that import them.

```
<parent>/
├── HippoRAG/          ← OSU-NLP-Group/HippoRAG
├── raptor/            ← parthsarthi03/raptor
├── ComoRAG/           ← EternityJune25/ComoRAG  (with patches/comorag/* applied)
├── arag/              ← Ayanami0730/arag
└── rag_test/          ← THIS REPO
```

Path resolution falls back through three tiers (see `utils/paths.py`):
1. environment variable (`HIPPORAG_PATH`, `RAPTOR_PATH`, `COMORAG_PATH`, `ARAG_PATH`)
2. `config/paths.yaml` (copy `paths.yaml.example` and edit)
3. sibling default (`../HippoRAG`, `../raptor`, …)

## Frozen versions

The numbers in `reports/RAG_COMPARISON_NARRATIVEQA.md` were produced against
these exact commit hashes. If you check out a different revision, regenerate
the comparison table with `scripts/build_comparison.py`.

| System | Upstream | Frozen commit | Notes |
|---|---|---|---|
| **HippoRAG 2** | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | `main` (untracked — see ⚠ below) | No patches required. Install with `pip install -e .` after cloning. |
| **RAPTOR** | [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) | `main` (untracked — see ⚠ below) | No patches required. Install with `pip install -e .` after cloning. |
| **ComoRAG** | [EternityJune25/ComoRAG](https://github.com/EternityJune25/ComoRAG) | `a4f8433` | **3 patches required** — see `patches/comorag/`. The patches make `VLLMOfflineOpenIE` import lazy + add a SBERT routing path. |
| **A-RAG** | [Ayanami0730/arag](https://github.com/Ayanami0730/arag) | `a44de6b` | No patches required. Build the SBERT sentence index once after cloning (`scripts/build_index.py`). |

> ⚠ **HippoRAG 2 and RAPTOR commit hashes are not frozen** in this snapshot —
> the wrapper code was developed against `main` at the time of writing
> (2026-05-08 ~ 2026-05-11). If you reproduce months later and upstream has
> introduced breaking API changes, the corresponding wrapper (`experiments/exp-003`
> or `exp-004`) may need a one-line import fix. The other three RAG systems
> (ComoRAG, A-RAG) are frozen to specific hashes and will reproduce exactly.
> To pin HippoRAG 2 / RAPTOR yourself, edit the `SYSTEMS=(...)` array in
> `scripts/setup_env.sh` and replace `main` with the desired commit hash.

## Setup (one-shot)

From the rag_test root:

```bash
bash scripts/setup_env.sh
```

This will:
1. Clone any missing sibling repos.
2. `git checkout` the frozen commit hashes above.
3. Apply patches for ComoRAG.
4. `pip install -e .` each system.

## Manual setup (without setup_env.sh)

```bash
PARENT=$(dirname "$PWD")        # if you’re in rag_test/

git clone https://github.com/OSU-NLP-Group/HippoRAG.git "$PARENT/HippoRAG"
git clone https://github.com/parthsarthi03/raptor.git  "$PARENT/raptor"
git clone https://github.com/EternityJune25/ComoRAG.git "$PARENT/ComoRAG"
git clone https://github.com/Ayanami0730/arag.git      "$PARENT/arag"

(cd "$PARENT/ComoRAG" && git checkout a4f8433)
(cd "$PARENT/arag"    && git checkout a44de6b)

# ComoRAG patches
(cd "$PARENT/ComoRAG" \
    && git apply ../rag_test/patches/comorag/01_lazy_import_vllm_openie.patch \
    && git apply ../rag_test/patches/comorag/02_embedding_model_init_sbert.patch \
    && cp ../rag_test/patches/comorag/SBERTEmbedding.py src/comorag/embedding_model/)

# Install
for d in HippoRAG raptor ComoRAG arag; do (cd "$PARENT/$d" && pip install -e .); done

# Common deps used by the wrappers + eval scorer
pip install numpy pandas pyyaml sentence-transformers tiktoken openai tqdm
```

## Verify

```bash
python utils/paths.py
```

Should print all four sibling paths with `[OK]` markers.

## Embeddings + data — generated, not vendored

The dense indices (HippoRAG KG, RAPTOR tree, ComoRAG memory pool, A-RAG sentence
embeddings) are produced on first run from the shared `reproduce/dataset/narrativeqa_dev_10_doc*.json`
files (already tracked in this repo). Expect ~10–40 min for the first-time
indexing of each system on CPU; subsequent runs reuse the cached indices.

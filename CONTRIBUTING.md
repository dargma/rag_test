# Contributing to rag_test

`rag_test/` is a small orchestrator that compares four RAG systems on a fixed
NarrativeQA split. Contributions that fit one of the categories below are
welcome.

## Scope

| In scope | Out of scope |
|---|---|
| Wrapper improvements for `exp-003` … `exp-006` | Architectural changes to upstream RAG repos (submit upstream) |
| Adding a 5th system as `exp-007-<system>/` | Forking upstream code into this repo (this repo is an orchestrator, not a vendor) |
| New patches for an upstream system → `patches/<system>/` | Patches that aren't needed for fair-comparison reproduction |
| Adding a new benchmark split → `reproduce/dataset/` + wrapper changes | Closed-source data or anything that breaks reproducibility |
| Documentation fixes (REPRODUCE.md, README.md, EXTERNAL_REPOS.md) | — |
| Metric additions to `experiments/eval_metrics.py` | Re-scoring published numbers using a different scorer (changes the comparison axis) |

## Workflow

1. Open an issue first if you're proposing a non-trivial change.
2. Fork → branch → PR. Keep PRs small and single-purpose.
3. Run `scripts/verify_results.py --tolerance 0.5` before submitting — your
   change must not regress any system's metrics beyond ±0.5 F1.
4. If you change a metric or evaluator behaviour, update `REPRODUCE.md` §2
   reference numbers + `scripts/verify_results.py` REFERENCE dict in the
   same PR.
5. If you patch an upstream system, document the patch in
   `EXTERNAL_REPOS.md` and add a `.patch` file under `patches/<system>/`.

## Local development

```bash
bash scripts/setup_env.sh           # one-shot install
export OPENAI_API_KEY=sk-...
bash scripts/run_all.sh             # rerun everything (~2.5 h on one A100)
python scripts/verify_results.py    # PR gate
```

For partial reruns:
```bash
WHICH="hipporag2 raptor" bash scripts/run_all.sh
python scripts/verify_results.py --which hipporag2 raptor
```

## Bug reports

Include in the report:
- `python --version` and `pip freeze | grep -E 'openai|sentence|tiktoken'`
- Output of `python utils/paths.py` (so we know which siblings resolved)
- Output of `python scripts/verify_results.py` showing which system / metric drifted
- The commit hash of each upstream repo (e.g. `git -C ../HippoRAG rev-parse HEAD`)

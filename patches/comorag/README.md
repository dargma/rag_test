# ComoRAG patches

Three modifications to upstream `EternityJune25/ComoRAG` required to run the
exp-005 benchmark with GPT-4.1-mini + SBERT (and to avoid pulling in vLLM
GPU deps unconditionally).

## Files

| File | Type | Target file in ComoRAG repo | Purpose |
|---|---|---|---|
| `01_lazy_import_vllm_openie.patch` | unified diff | `src/comorag/ComoRAG.py` | Make `VLLMOfflineOpenIE` import lazy (only when `openie_mode == "offline"`). Lets the module load without vLLM installed. |
| `02_embedding_model_init_sbert.patch` | unified diff | `src/comorag/embedding_model/__init__.py` | Route SBERT-style model names (`mpnet`, `minilm`, `sentence-transformers`, `multi-qa`, `all-…`) to the new `SBERTEmbeddingModel`. Make SBERT the unknown-name fallback (instead of returning `None`). |
| `SBERTEmbedding.py` | new file | `src/comorag/embedding_model/SBERTEmbedding.py` | Thin wrapper around `sentence-transformers.SentenceTransformer` matching ComoRAG's `BaseEmbeddingModel` API. |

## Apply

From a fresh clone of ComoRAG (sibling of `rag_test/`):

```bash
cd ../ComoRAG       # adjust if cloned elsewhere
git apply ../rag_test/patches/comorag/01_lazy_import_vllm_openie.patch
git apply ../rag_test/patches/comorag/02_embedding_model_init_sbert.patch
cp     ../rag_test/patches/comorag/SBERTEmbedding.py src/comorag/embedding_model/
```

Or use the orchestrator: `bash rag_test/scripts/setup_env.sh` (called from
`REPRODUCE.md` step 2).

## Why these patches are not contributed upstream

- The vLLM-lazy-import change is straightforward and would be a good PR; not
  contributed yet because we want the local repro to be reproducible from a
  known commit hash without depending on upstream PR merge timing.
- The SBERT routing is opinionated about which model-name substrings should
  resolve to SBERT vs OpenAI vs BGE — upstream would need a more general
  embedding-factory rework.

See `EXTERNAL_REPOS.md` for the upstream commit hash these patches apply against.

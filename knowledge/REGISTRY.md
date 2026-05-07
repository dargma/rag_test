# REGISTRY.md - Knowledge Base

## Established Facts

### HippoRAG2
- Knowledge graph-based RAG using OpenIE + PPR (Personalized PageRank) for retrieval
- NarrativeQA: 293 queries, 10 documents, 4111 corpus passages, F1=25.9 (Llama-3.3-70B)
- Supports vLLM (online/offline), OpenAI, Transformers backends; tested with Llama-3.1-8B-Instruct for local testing
- Embedding: nvidia/NV-Embed-v2; Graph type: facts_and_sim_passage_node_unidirectional

### RAPTOR
- Recursive tree-based RAG: cluster chunks via GMM+UMAP, summarize clusters, repeat to build tree
- NarrativeQA: ROUGE-L=30.8, BLEU-1=23.5, BLEU-4=6.4, METEOR=19.1 (with UnifiedQA-3B reader)
- Embedding: SBERT (multi-qa-mpnet-base-cos-v1), chunk size 100 tokens
- Collapsed tree retrieval with top-k, max 2000 tokens context

### NarrativeQA Dataset (HippoRAG2 version)
- 293 queries from 10 lengthy documents (movies/books)
- Corpus: 4111 passages, format: {idx, title, text}
- Answers: list of acceptable answer strings
- Available at: osunlp/HippoRAG_2 on HuggingFace

### Hardware
- 1x NVIDIA A100-SXM4-80GB (80GB VRAM)
- vLLM 0.6.6 with 45% GPU util (~36GB) + SBERT (~400MB) fits comfortably

### Experiment Results (2026-05-07) - Qwen2.5-7B (local vLLM)
- HippoRAG2: F1=16.94, EM=1.37, ROUGE-L=15.30, BLEU-1=12.96, BLEU-4=0.33, METEOR=25.86
- RAPTOR: F1=12.28, EM=0.34, ROUGE-L=11.05, BLEU-1=8.53, BLEU-4=0.62, METEOR=23.16
- HippoRAG2 wins 5/6 metrics

### Experiment Results (2026-05-07) - GPT-4.1-mini (OpenAI API)
- HippoRAG2: F1=24.72, EM=6.14, ROUGE-L=23.54, BLEU-1=19.85, BLEU-4=1.30, METEOR=31.06
- RAPTOR: F1=15.64, EM=0.00, ROUGE-L=14.16, BLEU-1=11.00, BLEU-4=0.62, METEOR=27.81
- HippoRAG2 wins ALL 6 metrics; F1 gap widens from +4.66 to +9.08 with stronger LLM
- GPT-4.1-mini HippoRAG2 F1=24.72 approaches paper's 70B result (F1=25.9)

## Hypotheses
- HippoRAG2's graph-based retrieval outperforms RAPTOR on NarrativeQA (confirmed, both LLM tiers)
- HippoRAG2's advantage grows with stronger LLMs (confirmed: +38% gap at 7B, +58% gap at GPT)
- GPT-4.1-mini is roughly equivalent to Llama-3.3-70B for this task (confirmed: F1 24.72 vs 25.9)

## Rejected Ideas
- RAPTOR's BLEU-4 advantage at 7B tier was spurious (disappeared with GPT-4.1-mini)

## Open Questions
- Would HippoRAG2 with single-step QA (no IRCoT) still outperform RAPTOR?
- How would GPT-4o (full size) compare — would it close the gap to the paper's 70B result?
- Can RAPTOR's tree structure be enhanced with entity-aware clustering?

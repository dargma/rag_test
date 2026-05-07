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

### Experiment Results (2026-05-07)
- LLM: Qwen/Qwen2.5-7B-Instruct via vLLM, Embedding: SBERT multi-qa-mpnet-base-cos-v1
- HippoRAG2: F1=16.94, EM=1.37, ROUGE-L=15.30, BLEU-1=12.96, BLEU-4=0.33, METEOR=25.86
- RAPTOR: F1=12.28, EM=0.34, ROUGE-L=11.05, BLEU-1=8.53, BLEU-4=0.62, METEOR=23.16
- HippoRAG2 wins 5/6 metrics; RAPTOR wins only BLEU-4 (both near zero)
- HippoRAG2 indexing slower (2345s vs 1458s) due to NER + triple extraction; RAPTOR QA faster (196s vs 844s)

## Hypotheses
- HippoRAG2's graph-based retrieval outperforms RAPTOR's tree-based retrieval on NarrativeQA (confirmed)
- Smaller 7B model reduces absolute scores vs 70B but preserves relative comparison validity
- RAPTOR's BLEU-4 advantage may be due to shorter, more precise answers vs HippoRAG2's verbose multi-step QA

## Rejected Ideas
(none yet)

## Open Questions
- How does 7B model quality compare to 70B for OpenIE extraction in HippoRAG2?
- Would RAPTOR perform better with larger context window (currently 3500 tokens)?
- HippoRAG2's multi-step QA (IRCoT) is slow — would single-step QA be competitive?

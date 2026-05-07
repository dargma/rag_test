# DECISIONS.md - Direction Change History

### 2026-05-07: Initial Experiment Design
- **Reason**: Compare HippoRAG2 vs RAPTOR on NarrativeQA benchmark
- **Before**: No experiment
- **After**: Use Llama-3.1-8B-Instruct (fits A100 80GB) instead of 70B; evaluate with F1, ROUGE-L, BLEU-1, BLEU-4, METEOR
- **Impact**: Both methods use same LLM for fair comparison; smaller model may reduce absolute scores but preserves relative comparison validity

### 2026-05-07: Model Change to Qwen2.5-7B-Instruct
- **Reason**: Llama-3.1-8B-Instruct is gated on HuggingFace (403 Forbidden), no HF token configured
- **Before**: Llama-3.1-8B-Instruct
- **After**: Qwen/Qwen2.5-7B-Instruct (ungated, 7B params, similar capability tier, supported by vLLM 0.6.6)
- **Impact**: Fair comparison preserved since both HippoRAG2 and RAPTOR use the same model

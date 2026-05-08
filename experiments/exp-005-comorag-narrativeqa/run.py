"""
exp-005: ComoRAG on NarrativeQA dev_10_doc with GPT-4.1-mini + SBERT
Fair comparison vs HippoRAG2 (exp-003) and RAPTOR (exp-004).

Setup:
- Dataset: narrativeqa_dev_10_doc (10 docs, 293 queries, 4111 passages)
- LLM: gpt-4.1-mini (OpenAI API)
- Embedding: SBERT (multi-qa-mpnet-base-cos-v1)
- Metrics: F1, EM, ROUGE-L, BLEU-1, BLEU-4, METEOR (via eval_metrics.py — same as exp-003/004)
"""
import os
import sys
import json
import time
import logging
import copy

# OpenAI API key
key_file = os.path.expanduser("~/.openai_key")
if os.path.exists(key_file):
    with open(key_file) as f:
        for line in f:
            if "OPENAI_API_KEY" in line and "=" in line:
                v = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["OPENAI_API_KEY"] = v
                break

assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

# Add ComoRAG src to path
sys.path.insert(0, "/content/drive/MyDrive/rag_test/ComoRAG")

from src.comorag.ComoRAG import ComoRAG
from src.comorag.utils.config_utils import BaseConfig
from src.comorag.utils.misc_utils import get_gold_answers

logging.basicConfig(level=logging.INFO)

# Paths
DATASET_DIR = "/content/local_fast/rag_eval/data/narrativeqa_dev_10_doc"
EMBEDDER_PATH = "/content/local_fast/rag_eval/embedder/multi-qa-mpnet-base-cos-v1"
WORKING_DIR = "/content/local_fast/rag_eval/outputs/comorag_narrativeqa"
RESULTS_DIR = "/content/drive/MyDrive/rag_test/experiments/exp-005-comorag-narrativeqa/results"

# Allow smoke test (subsample queries)
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
SMOKE_N = int(os.environ.get("SMOKE_N", "10"))


def main():
    # === Load corpus + qas ===
    corpus_path = os.path.join(DATASET_DIR, "corpus.jsonl")
    qas_path = os.path.join(DATASET_DIR, "qas.jsonl")

    with open(corpus_path) as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    docs = [doc["contents"] for doc in corpus]

    with open(qas_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if SMOKE_TEST:
        samples = samples[:SMOKE_N]
        print(f"[SMOKE] using first {SMOKE_N} queries")

    all_queries = [s["question"] for s in samples]
    print(f"Corpus: {len(docs)} passages")
    print(f"Queries: {len(all_queries)}")

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # === ComoRAG config ===
    config = BaseConfig(
        # LLM via OpenAI API (gpt-4.1-mini)
        llm_name="gpt-4.1-mini",
        llm_base_url=None,  # default OpenAI endpoint
        llm_api_key=os.environ["OPENAI_API_KEY"],
        # Embedding via local SBERT
        embedding_model_name=EMBEDDER_PATH,
        embedding_batch_size=64,
        # Dataset
        dataset="narrativeqa_dev_10_doc",
        # Output
        output_dir=os.path.join(WORKING_DIR, "result"),
        save_dir=WORKING_DIR,
        # ComoRAG-specific
        need_cluster=True,
        max_meta_loop_max_iterations=5,
        is_mc=False,  # not multi-choice (open-ended)
        max_tokens_ver=2000,
        max_tokens_sem=2000,
        max_tokens_epi=2000,
        # OpenIE / indexing
        openie_mode="online",
        max_new_tokens=2048,
        max_retry_attempts=10,
        corpus_len=len(corpus),
    )

    # === Index ===
    comorag = ComoRAG(global_config=config)

    t0 = time.time()
    comorag.index(docs)
    index_time = time.time() - t0
    print(f"\n[time] Indexing: {index_time:.1f}s")

    # === QA ===
    t0 = time.time()
    solutions = comorag.try_answer(all_queries)
    qa_time = time.time() - t0
    print(f"[time] QA: {qa_time:.1f}s")

    # === Attach gold answers ===
    gold_answers = get_gold_answers(samples)
    for idx, q in enumerate(solutions):
        q.gold_answers = list(gold_answers[idx])

    # === Save results ===
    def _extract_final_answer(text):
        """ComoRAG outputs long analysis ending with '### Final Answer\\n<short>'."""
        if "### Final Answer" in text:
            ans = text.split("### Final Answer", 1)[1].strip()
            ans = ans.lstrip(":.* \n")
            return ans.split("\n")[0].strip()
        return text

    results = []
    for idx, (q, sol) in enumerate(zip(all_queries, solutions)):
        raw = sol.answer if isinstance(sol.answer, str) else str(sol.answer)
        short = _extract_final_answer(raw)
        results.append({
            "id": idx,
            "question": q,
            "prediction": short,           # short answer for fair eval
            "prediction_full": raw,         # full ComoRAG output (long analysis)
            "gold_answers": list(sol.gold_answers),
        })

    suffix = "_smoke" if SMOKE_TEST else ""
    results_file = os.path.join(RESULTS_DIR, f"comorag_results{suffix}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[save] {results_file}")

    # === Eval with HippoRAG2's eval_metrics (FAIR) ===
    sys.path.insert(0, "/content/drive/MyDrive/rag_test/experiments")
    from eval_metrics import evaluate_all
    metrics = evaluate_all(
        [r["prediction"] for r in results],
        [r["gold_answers"] for r in results],
    )

    print("\n=== ComoRAG + GPT-4.1-mini Results on NarrativeQA ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    metrics["index_time_s"] = index_time
    metrics["qa_time_s"] = qa_time
    metrics["llm"] = "gpt-4.1-mini"
    metrics["embedding"] = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    metrics["dataset"] = "narrativeqa_dev_10_doc"
    metrics["n_queries"] = len(samples)
    metrics["n_passages"] = len(docs)

    metrics_file = os.path.join(RESULTS_DIR, f"comorag_metrics{suffix}.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] {metrics_file}")
    print(f"\nDone. Results → {RESULTS_DIR}")


if __name__ == "__main__":
    main()

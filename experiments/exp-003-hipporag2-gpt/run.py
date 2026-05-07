"""
exp-003: HippoRAG2 on NarrativeQA with GPT-4.1-mini
Uses OpenAI API (gpt-4.1-mini) + SBERT embedding (local)
"""
import os
import sys
import json
import time
import logging

sys.path.insert(0, '/content/drive/MyDrive/HippoRAG')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

logging.basicConfig(level=logging.INFO)

DATASET_NAME = "narrativeqa_dev_10_doc"
LLM_NAME = "gpt-4.1-mini"
EMBEDDING_NAME = "Transformers/sentence-transformers/multi-qa-mpnet-base-cos-v1"
SAVE_DIR = "/content/drive/MyDrive/HippoRAG/experiments/exp-003-hipporag2-gpt/results"
DATA_DIR = "/content/drive/MyDrive/HippoRAG/reproduce/dataset"

def main():
    corpus_path = os.path.join(DATA_DIR, f"{DATASET_NAME}_corpus.json")
    queries_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.json")

    with open(corpus_path) as f:
        corpus = json.load(f)
    with open(queries_path) as f:
        samples = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    all_queries = [s['question'] for s in samples]
    gold_answers = [s['answer'] for s in samples]

    print(f"Corpus: {len(docs)} passages")
    print(f"Queries: {len(all_queries)}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    config = BaseConfig(
        save_dir=SAVE_DIR,
        llm_base_url=None,  # Use default OpenAI endpoint
        llm_name=LLM_NAME,
        embedding_model_name=EMBEDDING_NAME,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        rerank_dspy_file_path=None,
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=64,
        max_new_tokens=2048,
        corpus_len=len(corpus),
        openie_mode="online",
        max_retry_attempts=10
    )

    hipporag = HippoRAG(global_config=config)

    t0 = time.time()
    hipporag.index(docs)
    index_time = time.time() - t0
    print(f"Indexing time: {index_time:.1f}s")

    t0 = time.time()
    qa_output = hipporag.rag_qa(
        queries=all_queries,
        gold_answers=[list(ga) for ga in gold_answers]
    )
    qa_time = time.time() - t0
    print(f"QA time: {qa_time:.1f}s")

    if isinstance(qa_output, tuple):
        queries_solutions = qa_output[0]
        predictions = [qs.answer for qs in queries_solutions]
        hipporag_qa_metrics = qa_output[-1] if len(qa_output) >= 5 else {}
        print(f"HippoRAG2 internal QA metrics: {hipporag_qa_metrics}")
    else:
        predictions = []
        hipporag_qa_metrics = {}

    results = []
    for i, (q, pred, gold) in enumerate(zip(all_queries, predictions, gold_answers)):
        results.append({
            "id": i,
            "question": q,
            "prediction": pred if isinstance(pred, str) else str(pred),
            "gold_answers": list(gold)
        })

    results_file = os.path.join(SAVE_DIR, "hipporag2_gpt_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    sys.path.insert(0, '/content/drive/MyDrive/HippoRAG/experiments')
    from eval_metrics import evaluate_all
    metrics = evaluate_all(
        [r["prediction"] for r in results],
        [r["gold_answers"] for r in results]
    )

    print("\n=== HippoRAG2 + GPT-4.1-mini Results on NarrativeQA ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    metrics["index_time_s"] = index_time
    metrics["qa_time_s"] = qa_time
    metrics["hipporag_internal"] = hipporag_qa_metrics
    with open(os.path.join(SAVE_DIR, "hipporag2_gpt_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()

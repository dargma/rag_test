"""
exp-002: RAPTOR on NarrativeQA
Uses Llama-3.1-8B-Instruct via vLLM (OpenAI-compatible) + SBERT for embeddings
Following RAPTOR paper: collapsed tree retrieval, 100-token chunks
"""
import os
import sys
import json
import time
import logging

sys.path.insert(0, '/content/drive/MyDrive/raptor')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from openai import OpenAI

from raptor import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)
from raptor.EmbeddingModels import SBertEmbeddingModel, BaseEmbeddingModel
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel

logging.basicConfig(level=logging.INFO)

# Configuration
LLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/content/drive/MyDrive/HippoRAG/reproduce/dataset"
DATASET_NAME = "narrativeqa_dev_10_doc"
SAVE_DIR = "/content/drive/MyDrive/HippoRAG/experiments/exp-002-raptor-narrativeqa/results"


class VLLMSummarizationModel(BaseSummarizationModel):
    """Summarization using local Llama via vLLM OpenAI-compatible API."""

    def __init__(self, model=LLM_MODEL, base_url=LLM_BASE_URL):
        self.client = OpenAI(api_key="dummy", base_url=base_url)
        self.model = model

    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Summarization error: {e}")
            return str(e)


class VLLMQAModel(BaseQAModel):
    """QA using local Llama via vLLM OpenAI-compatible API."""

    def __init__(self, model=LLM_MODEL, base_url=LLM_BASE_URL):
        self.client = OpenAI(api_key="dummy", base_url=base_url)
        self.model = model

    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Question Answering assistant. Answer concisely based on the given context."},
                    {"role": "user", "content": f"Given Context: {context}\n\nQuestion: {question}\n\nAnswer:"},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"QA error: {e}")
            return ""


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    corpus_path = os.path.join(DATA_DIR, f"{DATASET_NAME}_corpus.json")
    queries_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.json")

    with open(corpus_path) as f:
        corpus = json.load(f)
    with open(queries_path) as f:
        samples = json.load(f)

    all_queries = [s['question'] for s in samples]
    gold_answers = [s['answer'] for s in samples]

    # Group corpus by document (title prefix before first newline)
    # RAPTOR processes each document independently to build a tree
    # Combine all passages into a single text per document group
    doc_groups = {}
    for doc in corpus:
        title = doc['title']
        if title not in doc_groups:
            doc_groups[title] = []
        doc_groups[title].append(doc['text'])

    print(f"Documents: {len(doc_groups)}")
    print(f"Total passages: {len(corpus)}")
    print(f"Queries: {len(all_queries)}")

    # Build RAPTOR with custom models
    sbert_model = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    summarization_model = VLLMSummarizationModel()
    qa_model = VLLMQAModel()

    config = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        qa_model=qa_model,
        embedding_model=sbert_model,
        tb_max_tokens=100,          # 100 token chunks (paper setting)
        tb_num_layers=5,            # max tree layers
        tb_summarization_length=100,
        tr_top_k=10,                # retrieve top-k nodes
        tr_selection_mode="top_k",
    )

    # Build tree from all corpus documents combined
    # RAPTOR takes a single text string
    t0 = time.time()
    all_text = ""
    for title, texts in doc_groups.items():
        all_text += f"\n\n{title}\n" + "\n".join(texts)

    ra = RetrievalAugmentation(config=config)
    ra.add_documents(all_text)
    index_time = time.time() - t0
    print(f"Tree building time: {index_time:.1f}s")

    # Save tree
    tree_path = os.path.join(SAVE_DIR, "raptor_tree.pkl")
    ra.save(tree_path)

    # QA for each query
    t0 = time.time()
    predictions = []
    for i, question in enumerate(all_queries):
        if i % 50 == 0:
            print(f"Processing query {i}/{len(all_queries)}")
        try:
            answer = ra.answer_question(
                question=question,
                top_k=10,
                max_tokens=3500,
                collapse_tree=True,
            )
            predictions.append(answer if isinstance(answer, str) else str(answer))
        except Exception as e:
            print(f"Error on query {i}: {e}")
            predictions.append("")
    qa_time = time.time() - t0
    print(f"QA time: {qa_time:.1f}s")

    # Save results
    results = []
    for i, (q, pred, gold) in enumerate(zip(all_queries, predictions, gold_answers)):
        results.append({
            "id": i,
            "question": q,
            "prediction": pred,
            "gold_answers": list(gold)
        })

    results_file = os.path.join(SAVE_DIR, "raptor_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Evaluate
    sys.path.insert(0, '/content/drive/MyDrive/HippoRAG/experiments')
    from eval_metrics import evaluate_all
    metrics = evaluate_all(
        [r["prediction"] for r in results],
        [r["gold_answers"] for r in results]
    )

    print("\n=== RAPTOR Results on NarrativeQA ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    metrics["index_time_s"] = index_time
    metrics["qa_time_s"] = qa_time
    with open(os.path.join(SAVE_DIR, "raptor_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()

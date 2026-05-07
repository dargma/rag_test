"""
exp-004: RAPTOR on NarrativeQA with GPT-4.1-mini
Uses OpenAI API directly + SBERT for embeddings (local)
"""
import os
import sys
import json
import time
import logging

sys.path.insert(0, '/content/drive/MyDrive/raptor')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from openai import OpenAI

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.QAModels import BaseQAModel

logging.basicConfig(level=logging.INFO)

LLM_MODEL = "gpt-4.1-mini"
DATA_DIR = "/content/drive/MyDrive/HippoRAG/reproduce/dataset"
DATASET_NAME = "narrativeqa_dev_10_doc"
SAVE_DIR = "/content/drive/MyDrive/HippoRAG/experiments/exp-004-raptor-gpt/results"


class GPTSummarizationModel(BaseSummarizationModel):
    def __init__(self, model=LLM_MODEL):
        self.client = OpenAI()
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


class GPTQAModel(BaseQAModel):
    def __init__(self, model=LLM_MODEL):
        self.client = OpenAI()
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

    corpus_path = os.path.join(DATA_DIR, f"{DATASET_NAME}_corpus.json")
    queries_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.json")

    with open(corpus_path) as f:
        corpus = json.load(f)
    with open(queries_path) as f:
        samples = json.load(f)

    all_queries = [s['question'] for s in samples]
    gold_answers = [s['answer'] for s in samples]

    doc_groups = {}
    for doc in corpus:
        title = doc['title']
        if title not in doc_groups:
            doc_groups[title] = []
        doc_groups[title].append(doc['text'])

    print(f"Documents: {len(doc_groups)}")
    print(f"Total passages: {len(corpus)}")
    print(f"Queries: {len(all_queries)}")

    sbert_model = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")
    summarization_model = GPTSummarizationModel()
    qa_model = GPTQAModel()

    config = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        qa_model=qa_model,
        embedding_model=sbert_model,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_summarization_length=100,
        tr_top_k=10,
        tr_selection_mode="top_k",
    )

    t0 = time.time()
    all_text = ""
    for title, texts in doc_groups.items():
        all_text += f"\n\n{title}\n" + "\n".join(texts)

    ra = RetrievalAugmentation(config=config)
    ra.add_documents(all_text)
    index_time = time.time() - t0
    print(f"Tree building time: {index_time:.1f}s")

    ra.save(os.path.join(SAVE_DIR, "raptor_tree.pkl"))

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

    results = []
    for i, (q, pred, gold) in enumerate(zip(all_queries, predictions, gold_answers)):
        results.append({
            "id": i,
            "question": q,
            "prediction": pred,
            "gold_answers": list(gold)
        })

    results_file = os.path.join(SAVE_DIR, "raptor_gpt_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    sys.path.insert(0, '/content/drive/MyDrive/HippoRAG/experiments')
    from eval_metrics import evaluate_all
    metrics = evaluate_all(
        [r["prediction"] for r in results],
        [r["gold_answers"] for r in results]
    )

    print("\n=== RAPTOR + GPT-4.1-mini Results on NarrativeQA ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    metrics["index_time_s"] = index_time
    metrics["qa_time_s"] = qa_time
    with open(os.path.join(SAVE_DIR, "raptor_gpt_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()

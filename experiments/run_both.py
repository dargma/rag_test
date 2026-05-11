"""
Unified experiment: HippoRAG2 vs RAPTOR on NarrativeQA
Runs both systems sequentially with the same LLM and data.

Strategy:
  1. Start vLLM server → Run HippoRAG2 indexing + QA → Save results → Kill vLLM
  2. Start vLLM server → Run RAPTOR tree build + QA → Save results → Kill vLLM
  3. Evaluate both with unified metrics
"""
import os
import sys
import json
import time
import subprocess
import signal
import logging
from pathlib import Path

_HERE = Path(__file__).resolve().parent              # rag_test/experiments
sys.path.insert(0, str(_HERE.parent))                # rag_test root → utils.paths
from utils.paths import get_external_path, get_data_path   # noqa: E402
sys.path.insert(0, get_external_path("hipporag"))
sys.path.insert(0, get_external_path("raptor"))
sys.path.insert(0, str(_HERE))                       # rag_test/experiments  (eval_metrics)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====== Configuration ======
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_BASE_URL = "http://localhost:8000/v1"
EMBEDDING_NAME = "Transformers/sentence-transformers/multi-qa-mpnet-base-cos-v1"
DATASET_NAME = "narrativeqa_dev_10_doc"
DATA_DIR = get_data_path()                                            # rag_test/reproduce/dataset
HIPPO_SAVE_DIR  = str(_HERE / "exp-001-hipporag2-narrativeqa" / "results")
RAPTOR_SAVE_DIR = str(_HERE / "exp-002-raptor-narrativeqa"   / "results")


def start_vllm_server(gpu_util=0.5):
    """Start vLLM server and wait until ready."""
    logger.info("Starting vLLM server...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    proc = subprocess.Popen(
        ["python", "-m", "vllm.entrypoints.openai.api_server",
         "--model", LLM_MODEL,
         "--max-model-len", "4096",
         "--gpu-memory-utilization", str(gpu_util),
         "--port", "8000",
         "--dtype", "auto"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # Wait for server to be ready
    import urllib.request
    for i in range(60):
        time.sleep(10)
        try:
            req = urllib.request.urlopen(f"{LLM_BASE_URL}/models", timeout=5)
            data = json.loads(req.read())
            if data.get("data"):
                logger.info(f"vLLM server ready after {(i+1)*10}s")
                return proc
        except:
            pass
        logger.info(f"Waiting for vLLM... ({(i+1)*10}s)")

    proc.kill()
    raise RuntimeError("vLLM server failed to start")


def stop_vllm_server(proc):
    """Stop vLLM server and free GPU memory."""
    logger.info("Stopping vLLM server...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    time.sleep(5)
    # Force cleanup
    os.system("pkill -f 'vllm.entrypoints' 2>/dev/null")
    time.sleep(5)
    logger.info("vLLM server stopped")


def load_data():
    """Load NarrativeQA dataset."""
    with open(os.path.join(DATA_DIR, f"{DATASET_NAME}_corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(DATA_DIR, f"{DATASET_NAME}.json")) as f:
        samples = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    queries = [s['question'] for s in samples]
    gold_answers = [s['answer'] for s in samples]

    logger.info(f"Loaded {len(docs)} corpus docs, {len(queries)} queries")
    return corpus, docs, queries, gold_answers


def run_hipporag2(docs, queries, gold_answers):
    """Run HippoRAG2 indexing and QA."""
    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig

    os.makedirs(HIPPO_SAVE_DIR, exist_ok=True)

    config = BaseConfig(
        save_dir=HIPPO_SAVE_DIR,
        llm_base_url=LLM_BASE_URL,
        llm_name=LLM_MODEL,
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
        corpus_len=len(docs),
        openie_mode="online",
        max_retry_attempts=10,
    )

    hipporag = HippoRAG(global_config=config)

    # Indexing
    t0 = time.time()
    hipporag.index(docs)
    index_time = time.time() - t0
    logger.info(f"HippoRAG2 indexing time: {index_time:.1f}s")

    # QA
    t0 = time.time()
    qa_output = hipporag.rag_qa(
        queries=queries,
        gold_answers=[list(ga) for ga in gold_answers]
    )
    qa_time = time.time() - t0
    logger.info(f"HippoRAG2 QA time: {qa_time:.1f}s")

    # Extract predictions
    queries_solutions = qa_output[0]
    predictions = [qs.answer for qs in queries_solutions]
    internal_metrics = qa_output[-1] if len(qa_output) >= 5 else {}

    # Save
    results = []
    for i, (q, pred, gold) in enumerate(zip(queries, predictions, gold_answers)):
        results.append({
            "id": i, "question": q,
            "prediction": pred if isinstance(pred, str) else str(pred),
            "gold_answers": list(gold)
        })

    with open(os.path.join(HIPPO_SAVE_DIR, "hipporag2_results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results, index_time, qa_time, internal_metrics


def run_raptor(corpus, queries, gold_answers):
    """Run RAPTOR tree build and QA."""
    from openai import OpenAI
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    from raptor.EmbeddingModels import SBertEmbeddingModel
    from raptor.SummarizationModels import BaseSummarizationModel
    from raptor.QAModels import BaseQAModel

    os.makedirs(RAPTOR_SAVE_DIR, exist_ok=True)

    class VLLMSummarizer(BaseSummarizationModel):
        def __init__(self):
            self.client = OpenAI(api_key="dummy", base_url=LLM_BASE_URL)
        def summarize(self, context, max_tokens=500, stop_sequence=None):
            try:
                resp = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
                    ],
                    max_tokens=max_tokens, temperature=0, timeout=120
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.warning(f"Summarization error: {e}")
                return context[:500]

    class VLLMQAModel(BaseQAModel):
        def __init__(self):
            self.client = OpenAI(api_key="dummy", base_url=LLM_BASE_URL)
        def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
            try:
                resp = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a Question Answering assistant. Answer concisely based on the given context."},
                        {"role": "user", "content": f"Given Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
                    ],
                    max_tokens=max_tokens, temperature=0, timeout=60
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"QA error: {e}")
                return ""

    sbert = SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1")

    config = RetrievalAugmentationConfig(
        summarization_model=VLLMSummarizer(),
        qa_model=VLLMQAModel(),
        embedding_model=sbert,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_summarization_length=100,
        tr_top_k=10,
        tr_selection_mode="top_k",
    )

    # Build tree from all corpus text
    doc_groups = {}
    for doc in corpus:
        title = doc['title']
        if title not in doc_groups:
            doc_groups[title] = []
        doc_groups[title].append(doc['text'])

    all_text = ""
    for title, texts in doc_groups.items():
        all_text += f"\n\n{title}\n" + "\n".join(texts)

    t0 = time.time()
    ra = RetrievalAugmentation(config=config)
    ra.add_documents(all_text)
    index_time = time.time() - t0
    logger.info(f"RAPTOR tree build time: {index_time:.1f}s")

    # Save tree
    ra.save(os.path.join(RAPTOR_SAVE_DIR, "raptor_tree.pkl"))

    # QA
    t0 = time.time()
    predictions = []
    for i, question in enumerate(queries):
        if i % 50 == 0:
            logger.info(f"RAPTOR QA: {i}/{len(queries)}")
        try:
            answer = ra.answer_question(
                question=question, top_k=10, max_tokens=3500, collapse_tree=True
            )
            predictions.append(answer if isinstance(answer, str) else str(answer))
        except Exception as e:
            logger.warning(f"RAPTOR QA error on query {i}: {e}")
            predictions.append("")
    qa_time = time.time() - t0
    logger.info(f"RAPTOR QA time: {qa_time:.1f}s")

    # Save
    results = []
    for i, (q, pred, gold) in enumerate(zip(queries, predictions, gold_answers)):
        results.append({
            "id": i, "question": q,
            "prediction": pred, "gold_answers": list(gold)
        })

    with open(os.path.join(RAPTOR_SAVE_DIR, "raptor_results.json"), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results, index_time, qa_time


def evaluate_and_report(hippo_results, raptor_results, hippo_times, raptor_times):
    """Evaluate both and create comparison report."""
    from eval_metrics import evaluate_all

    hippo_metrics = evaluate_all(
        [r["prediction"] for r in hippo_results],
        [r["gold_answers"] for r in hippo_results]
    )
    raptor_metrics = evaluate_all(
        [r["prediction"] for r in raptor_results],
        [r["gold_answers"] for r in raptor_results]
    )

    hippo_metrics["index_time_s"] = hippo_times[0]
    hippo_metrics["qa_time_s"] = hippo_times[1]
    raptor_metrics["index_time_s"] = raptor_times[0]
    raptor_metrics["qa_time_s"] = raptor_times[1]

    with open(os.path.join(HIPPO_SAVE_DIR, "hipporag2_metrics.json"), 'w') as f:
        json.dump(hippo_metrics, f, indent=2)
    with open(os.path.join(RAPTOR_SAVE_DIR, "raptor_metrics.json"), 'w') as f:
        json.dump(raptor_metrics, f, indent=2)

    print("\n" + "="*70)
    print("  HippoRAG2 vs RAPTOR on NarrativeQA - Final Results")
    print("="*70)
    print(f"\n{'Metric':<15} {'HippoRAG2':>12} {'RAPTOR':>12} {'Delta':>12}")
    print("-"*55)
    for metric in ["F1", "EM", "ROUGE-L", "BLEU-1", "BLEU-4", "METEOR"]:
        h = hippo_metrics[metric]
        r = raptor_metrics[metric]
        d = h - r
        winner = "+" if d > 0 else "-" if d < 0 else "="
        print(f"{metric:<15} {h:>11.2f}% {r:>11.2f}% {d:>+11.2f} {winner}")
    print("-"*55)
    print(f"{'Index Time':<15} {hippo_metrics['index_time_s']:>10.0f}s {raptor_metrics['index_time_s']:>10.0f}s")
    print(f"{'QA Time':<15} {hippo_metrics['qa_time_s']:>10.0f}s {raptor_metrics['qa_time_s']:>10.0f}s")
    print("="*70)

    return hippo_metrics, raptor_metrics


def main():
    logger.info("="*50)
    logger.info("Starting HippoRAG2 vs RAPTOR comparison on NarrativeQA")
    logger.info("="*50)

    # Disk check
    usage = int(os.popen("df -h . | awk 'NR==2 {print $5}' | tr -d '%'").read().strip())
    if usage >= 95:
        raise RuntimeError("DISK 95% - All operations halted. Cleanup required.")

    corpus, docs, queries, gold_answers = load_data()

    # ====== Phase 1: HippoRAG2 ======
    logger.info("="*50)
    logger.info("Phase 1: Running HippoRAG2")
    logger.info("="*50)

    vllm_proc = start_vllm_server(gpu_util=0.5)
    try:
        hippo_results, hippo_idx_time, hippo_qa_time, hippo_internal = run_hipporag2(docs, queries, gold_answers)
    finally:
        stop_vllm_server(vllm_proc)

    # ====== Phase 2: RAPTOR ======
    logger.info("="*50)
    logger.info("Phase 2: Running RAPTOR")
    logger.info("="*50)

    vllm_proc = start_vllm_server(gpu_util=0.5)
    try:
        raptor_results, raptor_idx_time, raptor_qa_time = run_raptor(corpus, queries, gold_answers)
    finally:
        stop_vllm_server(vllm_proc)

    # ====== Phase 3: Evaluate ======
    logger.info("="*50)
    logger.info("Phase 3: Evaluation")
    logger.info("="*50)

    hippo_metrics, raptor_metrics = evaluate_and_report(
        hippo_results, raptor_results,
        (hippo_idx_time, hippo_qa_time),
        (raptor_idx_time, raptor_qa_time)
    )

    logger.info("All done!")


if __name__ == "__main__":
    main()

"""Convert rag_test's NarrativeQA dev_10_doc data into A-RAG's expected format.

Inputs (rag_test repo):
    reproduce/dataset/narrativeqa_dev_10_doc.json         — 293 queries
    reproduce/dataset/narrativeqa_dev_10_doc_corpus.json  — 4111 chunks (140-tok)

Outputs (under <arag_repo>/data/narrativeqa/, where arag_repo resolves via utils.paths):
    chunks.json     — list of {id, title, text}                (renamed `idx` → `id`)
    questions.json  — list of {idx, question, answer: [str,…]}

This is idempotent — re-running overwrites the destination files.
"""
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))      # rag_test root
from utils.paths import get_external_path, get_data_path  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arag_path", default=None,
                    help="Override A-RAG repo location (default: sibling resolution).")
    ap.add_argument("--out_subdir", default="data/narrativeqa",
                    help="Subdir inside A-RAG repo to write chunks/questions to.")
    args = ap.parse_args()

    arag_root = Path(args.arag_path or get_external_path("arag"))
    out_dir = arag_root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_src = Path(get_data_path("narrativeqa_dev_10_doc_corpus.json"))
    queries_src = Path(get_data_path("narrativeqa_dev_10_doc.json"))
    if not corpus_src.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_src}")
    if not queries_src.exists():
        raise FileNotFoundError(f"queries not found: {queries_src}")

    # ----- chunks -----
    corpus = json.load(open(corpus_src))
    if not isinstance(corpus, list):
        raise ValueError(f"corpus must be a list, got {type(corpus)}")
    chunks = []
    for c in corpus:
        # source has either 'idx' (HippoRAG2 convention) or 'id'
        cid = c.get("id") or c.get("idx")
        if cid is None:
            raise KeyError(f"chunk missing id/idx: {list(c.keys())[:5]}")
        chunks.append({
            "id":    cid,
            "title": c.get("title", ""),
            "text":  c.get("text", c.get("contents", "")),
        })
    json.dump(chunks, open(out_dir / "chunks.json", "w"), ensure_ascii=False)
    print(f"  wrote {len(chunks):4d} chunks → {out_dir / 'chunks.json'}")

    # ----- questions -----
    queries = json.load(open(queries_src))
    if isinstance(queries, dict) and "data" in queries:
        queries = queries["data"]
    if not isinstance(queries, list):
        raise ValueError(f"queries must be a list, got {type(queries)}")
    out_q = []
    for i, q in enumerate(queries):
        # HippoRAG2 dev_10_doc uses 'question' + 'answer' (list of 2 reference strs)
        question = q.get("question") or q.get("query")
        answers = q.get("answer") or q.get("answers") or q.get("gold_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        out_q.append({"idx": i, "question": question, "answer": answers})
    json.dump(out_q, open(out_dir / "questions.json", "w"), ensure_ascii=False)
    print(f"  wrote {len(out_q):4d} questions → {out_dir / 'questions.json'}")

    print("\nNext step: build the sentence-level dense index (one-time, ~2 min on CPU):")
    print(f"  cd {arag_root}")
    print(f"  python scripts/build_index.py --chunks {args.out_subdir}/chunks.json --out_dir {args.out_subdir}/index")


if __name__ == "__main__":
    main()

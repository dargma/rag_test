"""Eval A-RAG predictions.jsonl on NarrativeQA with the same scorer as
HippoRAG2 / RAPTOR / ComoRAG (max-over-2-references F1/EM/ROUGE-L/BLEU/METEOR).

Usage::

    python eval.py [predictions.jsonl] [eval_summary.json]

Defaults: reads `results/predictions.jsonl` (next to this script) and writes
`results/eval_summary.json`. Override either by passing CLI args.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))            # rag_test/experiments  (eval_metrics)
sys.path.insert(0, str(_HERE))                   # rag_test/experiments/exp-006-...  (clean_answer)
from eval_metrics import evaluate_all  # type: ignore
from clean_answer import clean_answer  # type: ignore


def main():
    default_pred = _HERE / "results" / "predictions.jsonl"
    default_out  = _HERE / "results" / "eval_summary.json"
    pred_path = sys.argv[1] if len(sys.argv) > 1 else str(default_pred)
    out_path  = sys.argv[2] if len(sys.argv) > 2 else str(default_out)

    rows = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} predictions from {pred_path}")

    raw_preds, clean_preds = [], []
    gold_lists = []
    skipped = 0
    for r in rows:
        pred = r.get("pred_answer") or ""
        golds = r.get("gold_answer") or []
        if not isinstance(golds, list):
            golds = [golds]
        if not golds:
            skipped += 1
            continue
        raw_preds.append(pred)
        clean_preds.append(clean_answer(pred))
        gold_lists.append(golds)
    print(f"  pred (non-empty gold): {len(raw_preds)}, skipped: {skipped}")

    metrics_raw = evaluate_all(raw_preds, gold_lists)
    metrics_clean = evaluate_all(clean_preds, gold_lists)
    metrics = {"raw": metrics_raw, "cleaned": metrics_clean, "N": len(raw_preds)}

    # Aux stats from agent run
    aux = {}
    if rows:
        loops = [r.get("loops", 0) for r in rows if isinstance(r.get("loops"), int)]
        toks = [r.get("total_retrieved_tokens", 0) for r in rows if isinstance(r.get("total_retrieved_tokens"), int)]
        chunks = [r.get("chunks_read_count", 0) for r in rows if isinstance(r.get("chunks_read_count"), int)]
        costs = [r.get("total_cost", 0.0) for r in rows if isinstance(r.get("total_cost"), (int, float))]
        raw_lens = [len(p.split()) for p in raw_preds]
        clean_lens = [len(p.split()) for p in clean_preds]
        if loops:  aux["avg_loops"] = float(sum(loops) / len(loops))
        if toks:   aux["avg_retrieved_tokens"] = float(sum(toks) / len(toks))
        if chunks: aux["avg_chunks_read"] = float(sum(chunks) / len(chunks))
        if costs:
            aux["total_cost"] = float(sum(costs))
            aux["avg_cost"] = float(sum(costs) / len(costs))
        aux["avg_pred_len_raw"] = float(sum(raw_lens) / len(raw_lens))
        aux["avg_pred_len_clean"] = float(sum(clean_lens) / len(clean_lens))
    metrics["aux"] = aux

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")

    keys = ["F1", "EM", "ROUGE-L", "BLEU-1", "BLEU-4", "METEOR"]
    print("\n=== A-RAG on NarrativeQA (fair-matched: SBERT + GPT-4.1-mini) ===")
    print(f"N = {metrics['N']}")
    print(f"\n{'metric':<10s}{'raw':>10s}{'cleaned':>10s}")
    print("-"*30)
    for k in keys:
        rv = metrics_raw.get(k, 0)
        cv = metrics_clean.get(k, 0)
        print(f"{k:<10s}{rv:>10.2f}{cv:>10.2f}")
    print("\n--- aux ---")
    for k, v in aux.items():
        print(f"  {k:24s}: {v:.3f}" if isinstance(v, float) else f"  {k:24s}: {v}")


if __name__ == "__main__":
    main()

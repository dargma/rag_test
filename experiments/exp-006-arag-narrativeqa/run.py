"""A-RAG runner for NarrativeQA dev_10_doc (293 queries).

Pipeline:
    1) build_data.py   — convert rag_test's narrativeqa corpus/queries into A-RAG's
                         {chunks,questions}.json format under <arag>/data/narrativeqa/
    2) <arag>/scripts/build_index.py — build sentence-level SBERT index (one-time)
    3) <arag>/scripts/batch_runner.py — run the agent over all 293 queries
    4) eval.py         — score predictions with the same metric module as the
                         other 3 RAG systems and write eval_summary.json

Outputs land in `experiments/exp-006-arag-narrativeqa/results/`:
    predictions.jsonl
    eval_summary.json

Run::

    python experiments/exp-006-arag-narrativeqa/run.py [--skip_build_index] [--workers N]

Requires `OPENAI_API_KEY` (also re-exported as `ARAG_API_KEY` for the A-RAG client).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))   # rag_test root
from utils.paths import get_external_path      # type: ignore


def _run(cmd: list[str], *, env: dict | None = None, cwd: str | None = None):
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env, cwd=cwd)
    if res.returncode != 0:
        sys.exit(f"command failed (rc={res.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_build_data",  action="store_true",
                    help="Skip the chunks/questions conversion (use existing files in <arag>/data/narrativeqa)")
    ap.add_argument("--skip_build_index", action="store_true",
                    help="Skip SBERT index build (use existing sentence_index.pkl)")
    ap.add_argument("--workers", type=int, default=5,
                    help="Concurrent worker processes for the agent loop (default 5)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N queries (smoke test). 0 = run all 293.")
    ap.add_argument("--config", default=str(_HERE / "configs" / "narrativeqa.yaml"))
    args = ap.parse_args()

    arag = Path(get_external_path("arag"))
    py = sys.executable
    results_dir = _HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # API key sanity
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ARAG_API_KEY")
    if not api_key:
        sys.exit("ERROR: set OPENAI_API_KEY (or ARAG_API_KEY) before running.")
    env = {**os.environ, "OPENAI_API_KEY": api_key, "ARAG_API_KEY": api_key,
           "ARAG_BASE_URL": os.environ.get("ARAG_BASE_URL", "https://api.openai.com/v1"),
           "ARAG_MODEL": os.environ.get("ARAG_MODEL", "gpt-4.1-mini")}

    # 1) build_data — write chunks/questions into A-RAG repo
    if not args.skip_build_data:
        _run([py, str(_HERE / "build_data.py")], env=env)

    chunks_p = arag / "data" / "narrativeqa" / "chunks.json"
    index_p  = arag / "data" / "narrativeqa" / "index" / "sentence_index.pkl"

    # 2) sentence index (SBERT) — build once, reuse
    if not args.skip_build_index and not index_p.exists():
        _run([py, str(arag / "scripts" / "build_index.py"),
              "--chunks", str(chunks_p),
              "--out_dir", str(index_p.parent)], env=env, cwd=str(arag))
    else:
        print(f"[info] using existing index at {index_p}")

    # 3) batch_runner — write predictions into experiment-local results/
    cmd = [py, str(arag / "scripts" / "batch_runner.py"),
           "--config", args.config,
           "--questions", str(arag / "data" / "narrativeqa" / "questions.json"),
           "--output", str(results_dir),
           "--workers", str(args.workers)]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    _run(cmd, env=env, cwd=str(arag))

    # 4) eval
    pred = results_dir / "predictions.jsonl"
    out  = results_dir / "eval_summary.json"
    _run([py, str(_HERE / "eval.py"), str(pred), str(out)], env=env)

    print(f"\n✓ A-RAG NarrativeQA run complete. results/ → {results_dir}")


if __name__ == "__main__":
    main()

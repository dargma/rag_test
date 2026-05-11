#!/usr/bin/env python3
"""Verify reproduced results against the reference values in REPRODUCE.md.

Usage:
    python scripts/verify_results.py                  # check all 4 systems
    python scripts/verify_results.py --tolerance 0.5  # allow ±0.5 F1
    python scripts/verify_results.py --which hipporag2 raptor

Exits with code 0 if all metrics match within tolerance, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reference metrics from canonical 2026-05-08 reproduction
# Source: experiments/exp-00X/results/*_metrics.json (gpt-4.1-mini, NarrativeQA dev_10_doc, 293 queries)
REFERENCE = {
    "hipporag2": {
        "result_path": "experiments/exp-003-hipporag2-gpt/results/hipporag2_gpt_metrics.json",
        "F1":     24.72,
        "EM":      6.14,
        "ROUGE-L": 23.54,
        "BLEU-1":  19.85,
        "BLEU-4":   1.30,
        "METEOR":  31.06,
    },
    "raptor": {
        "result_path": "experiments/exp-004-raptor-gpt/results/raptor_gpt_metrics.json",
        "F1":     15.64,
        "EM":      0.00,
        "ROUGE-L": 14.16,
        "BLEU-1":  11.00,
        "BLEU-4":   0.62,
        "METEOR":  27.81,
    },
    "comorag": {
        "result_path": "experiments/exp-005-comorag-narrativeqa/results/comorag_metrics.json",
        "F1":     23.09,
        "EM":      8.53,
        "ROUGE-L": 22.37,
        "BLEU-1":  19.77,
        "BLEU-4":   0.53,
        "METEOR":  24.39,
    },
    "arag_raw": {
        "result_path": "experiments/exp-006-arag-narrativeqa/results/eval_summary.json",
        "nested_key": "raw",  # value is nested under summary["raw"]
        "F1":      6.61,
        "EM":      0.00,
        "ROUGE-L": 5.84,
        "BLEU-1":  3.71,
        "BLEU-4":  0.17,
        "METEOR": 18.94,
    },
    "arag_cleaned": {
        "result_path": "experiments/exp-006-arag-narrativeqa/results/eval_summary.json",
        "nested_key": "cleaned",
        "F1":     12.07,
        "EM":      0.00,
        "ROUGE-L": 10.44,
        "BLEU-1":  7.84,
        "BLEU-4":  0.41,
        "METEOR": 24.31,
    },
}


def load_metrics(spec):
    p = REPO_ROOT / spec["result_path"]
    if not p.exists():
        return None, f"result file missing: {p}"
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        return None, f"could not parse {p}: {e}"
    nested = spec.get("nested_key")
    if nested:
        if nested not in data:
            return None, f"key '{nested}' not in {p}"
        data = data[nested]
    return data, None


def fmt(v):
    return "—" if v is None else f"{v:6.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="absolute deviation allowed per metric (default 0.5)")
    ap.add_argument("--which", nargs="+", default=None,
                    choices=list(REFERENCE.keys()),
                    help="subset of systems to verify (default: all)")
    args = ap.parse_args()

    systems = args.which or list(REFERENCE.keys())
    metric_keys = ["F1", "EM", "ROUGE-L", "BLEU-1", "BLEU-4", "METEOR"]

    print(f"Tolerance: ±{args.tolerance:.2f} (absolute)")
    print()
    header = f"{'system':<14} | " + " | ".join(f"{m:>15}" for m in metric_keys)
    print(header)
    print("-" * len(header))

    all_ok = True
    for sys_name in systems:
        spec = REFERENCE[sys_name]
        actual, err = load_metrics(spec)
        if err:
            print(f"{sys_name:<14} | ERROR: {err}")
            all_ok = False
            continue
        cells = []
        for m in metric_keys:
            ref = spec[m]
            act = actual.get(m)
            if act is None:
                cells.append(f" {fmt(None):>6} (---)")
                continue
            delta = act - ref
            ok = abs(delta) <= args.tolerance
            mark = "✓" if ok else "✗"
            cells.append(f"{fmt(act)} ({delta:+.2f}){mark}")
            if not ok:
                all_ok = False
        row = f"{sys_name:<14} | " + " | ".join(f"{c:>15}" for c in cells)
        print(row)

    print()
    if all_ok:
        print("✓ All systems within tolerance of reference values.")
        sys.exit(0)
    else:
        print("✗ At least one metric outside tolerance. Investigate or rerun.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Unified evaluation metrics for NarrativeQA: F1, ROUGE-L, BLEU-1, BLEU-4, METEOR
Used for both HippoRAG2 and RAPTOR evaluation.
"""

import json
import re
import string
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    lcs = _lcs_length(pred_tokens, gt_tokens)
    precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _count_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu_n(prediction: str, ground_truth: str, n: int) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) < n or len(gt_tokens) < n:
        return 0.0
    pred_ngrams = _count_ngrams(pred_tokens, n)
    gt_ngrams = _count_ngrams(gt_tokens, n)
    clipped = {ng: min(count, gt_ngrams.get(ng, 0)) for ng, count in pred_ngrams.items()}
    numerator = sum(clipped.values())
    denominator = sum(pred_ngrams.values())
    if denominator == 0:
        return 0.0
    # Modified precision (no brevity penalty for single reference comparison)
    bp = min(1.0, np.exp(1 - len(gt_tokens) / max(len(pred_tokens), 1)))
    return bp * numerator / denominator


def compute_meteor(prediction: str, ground_truth: str) -> float:
    """Simplified METEOR: unigram precision/recall with harmonic mean (alpha=0.9)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    matches = sum((pred_counter & gt_counter).values())

    precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = matches / len(gt_tokens) if len(gt_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    alpha = 0.9  # weight recall more
    fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    return fmean


def evaluate_all(predictions: List[str], gold_answers_list: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate predictions against gold answers with all metrics.
    For each sample, take max score across gold answers.
    """
    assert len(predictions) == len(gold_answers_list)

    metrics = {"F1": [], "EM": [], "ROUGE-L": [], "BLEU-1": [], "BLEU-4": [], "METEOR": []}

    for pred, golds in zip(predictions, gold_answers_list):
        if not pred:
            pred = ""
        f1_scores = [compute_f1(pred, g) for g in golds]
        em_scores = [compute_exact_match(pred, g) for g in golds]
        rouge_scores = [compute_rouge_l(pred, g) for g in golds]
        bleu1_scores = [compute_bleu_n(pred, g, 1) for g in golds]
        bleu4_scores = [compute_bleu_n(pred, g, 4) for g in golds]
        meteor_scores = [compute_meteor(pred, g) for g in golds]

        metrics["F1"].append(max(f1_scores))
        metrics["EM"].append(max(em_scores))
        metrics["ROUGE-L"].append(max(rouge_scores))
        metrics["BLEU-1"].append(max(bleu1_scores))
        metrics["BLEU-4"].append(max(bleu4_scores))
        metrics["METEOR"].append(max(meteor_scores))

    return {k: np.mean(v) * 100 for k, v in metrics.items()}


def evaluate_from_file(results_path: str) -> Dict[str, float]:
    """Load results JSON and compute metrics."""
    with open(results_path) as f:
        results = json.load(f)

    predictions = [r["prediction"] for r in results]
    gold_answers = [r["gold_answers"] for r in results]

    return evaluate_all(predictions, gold_answers)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        metrics = evaluate_from_file(sys.argv[1])
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")

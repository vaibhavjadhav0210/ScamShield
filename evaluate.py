"""Evaluation script comparing RuleDetector, MLDetector, and HybridDetector.

Evaluates all three detectors on the held-out 20% stratified test split and
saves:
    - results/metrics.json
    - results/confusion_matrix.png
    - results/comparison.png
    - results/misclassified.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from scamshield.detectors.base import Detector
from scamshield.detectors.hybrid import HybridDetector
from scamshield.detectors.ml import DEFAULT_MODEL_PATH, MLDetector
from scamshield.detectors.rule import RuleDetector
from scamshield.models.message import SCAM_CATEGORIES, MessageFactory

DATA_PATH = Path("data") / "messages.csv"
RESULTS_DIR = Path("results")


def _evaluate_detector(
    detector: Detector,
    texts: list[str],
    msg_types: list[str],
    y_true: list[str],
) -> tuple[dict[str, float], list[str], list[int]]:
    """Run `detector` over the test set and compute classification metrics."""
    y_pred: list[str] = []
    risk_scores: list[int] = []

    for text, mtype in zip(texts, msg_types):
        msg = MessageFactory.create(mtype, text)
        res = detector.detect(msg)
        y_pred.append(res.scam_type)
        risk_scores.append(res.risk_score)

    y_true_bin = ["SAFE" if y == "Safe" else "SCAM" for y in y_true]
    y_pred_bin = ["SAFE" if y == "Safe" else "SCAM" for y in y_pred]

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "binary_accuracy": round(float(accuracy_score(y_true_bin, y_pred_bin)), 4),
        "binary_f1_scam": round(
            float(f1_score(y_true_bin, y_pred_bin, pos_label="SCAM", zero_division=0)), 4
        ),
    }
    return metrics, y_pred, risk_scores


def _plot_confusion_matrix(y_true: list[str], y_pred: list[str], out_path: Path) -> None:
    """Save a confusion matrix heatmap for HybridDetector."""
    labels = list(SCAM_CATEGORIES)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title("ScamShield HybridDetector Confusion Matrix (Test Split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_comparison(all_metrics: dict[str, dict[str, float]], out_path: Path) -> None:
    """Save a grouped bar chart comparing RuleDetector, MLDetector, and HybridDetector."""
    detectors = list(all_metrics.keys())
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, det_name in enumerate(detectors):
        vals = [all_metrics[det_name][k] for k in metric_keys]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=det_name)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Detector Comparison on Held-Out Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_misclassified(
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    risk_scores: list[int],
    out_path: Path,
    max_examples: int = 10,
) -> None:
    """Write a few misclassified test examples for error analysis."""
    lines: list[str] = ["Misclassified Examples (HybridDetector)", "=" * 50, ""]
    count = 0
    for text, true_lbl, pred_lbl, score in zip(texts, y_true, y_pred, risk_scores):
        if true_lbl != pred_lbl:
            count += 1
            lines.append(
                f"[{count}] True: {true_lbl} | Predicted: {pred_lbl} | Risk: {score}/100"
            )
            lines.append(f"    Text: {text[:180]}")
            lines.append("")
            if count >= max_examples:
                break
    if count == 0:
        lines.append("No misclassified examples on the held-out test split!")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(
    data_path: Path = DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, dict[str, float]]:
    """Run full comparison and write metrics, charts, and error examples."""
    df = pd.read_csv(data_path)
    _X_train, X_test, _y_train, y_test, _m_train, m_test = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        df["message_type"].astype(str),
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    test_texts = X_test.tolist()
    test_types = m_test.tolist()
    test_labels = y_test.tolist()

    rule_det = RuleDetector()
    ml_det = MLDetector(model_path=model_path)
    hybrid_det = HybridDetector(ml_detector=ml_det, rule_detector=rule_det)

    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict[str, float]] = {}

    hybrid_preds: list[str] = []
    hybrid_scores: list[int] = []

    for det in (rule_det, ml_det, hybrid_det):
        metrics, preds, scores = _evaluate_detector(det, test_texts, test_types, test_labels)
        all_metrics[det.name] = metrics
        if isinstance(det, HybridDetector):
            hybrid_preds = preds
            hybrid_scores = scores

    (results_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    _plot_confusion_matrix(test_labels, hybrid_preds, results_dir / "confusion_matrix.png")
    _plot_comparison(all_metrics, results_dir / "comparison.png")
    _save_misclassified(
        test_texts,
        test_labels,
        hybrid_preds,
        hybrid_scores,
        results_dir / "misclassified.txt",
    )

    print("Evaluation complete. Metrics summary:")
    for name, m in all_metrics.items():
        print(
            f"  {name:15s} -> Acc: {m['accuracy']:.4f} | "
            f"Prec: {m['precision_macro']:.4f} | "
            f"Rec: {m['recall_macro']:.4f} | "
            f"F1: {m['f1_macro']:.4f}"
        )
    return all_metrics


if __name__ == "__main__":
    run_evaluation()
